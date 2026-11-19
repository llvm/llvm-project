# Rules for checking the binary size of LLVM libc entrypoints.
#
# These checks only build and inspect objects; they never execute target code,
# so they work for cross and baremetal builds. They are enabled with
# LIBC_INCLUDE_SIZE_CHECKS, independently of LLVM_INCLUDE_TESTS. See
# libc/utils/size_check/README.md for details.

# Finds an LLVM binary utility, preferring the one next to the compiler.
function(_libc_size_check_find_tool output_var tool)
  if(${output_var})
    return()
  endif()
  set(candidate "")
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -print-prog-name=${tool}
    OUTPUT_VARIABLE candidate
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE result
    ERROR_QUIET
  )
  if(result EQUAL 0 AND IS_ABSOLUTE "${candidate}" AND EXISTS "${candidate}")
    set(${output_var} "${candidate}" CACHE FILEPATH "Path to ${tool}")
    return()
  endif()
  get_filename_component(compiler_dir ${CMAKE_CXX_COMPILER} DIRECTORY)
  find_program(
    ${output_var}
    NAMES ${tool} ${tool}-${LLVM_VERSION_MAJOR}
    HINTS ${compiler_dir} ${LLVM_TOOLS_DIR}
  )
endfunction()

# Writes the predefined macros of the compiler for the given compile options to
# `output_file`. The size checker uses them to select target specific rules.
function(_libc_size_check_dump_macros output_file)
  set(cmd ${CMAKE_CXX_COMPILER})
  if(CMAKE_CXX_COMPILER_TARGET)
    list(APPEND cmd "--target=${CMAKE_CXX_COMPILER_TARGET}")
  endif()
  separate_arguments(cxx_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
  _libc_size_check_expand_options(options ${ARGN})
  list(APPEND cmd ${cxx_flags} ${options})
  list(APPEND cmd -dM -E -x c++ /dev/null)
  get_filename_component(output_dir ${output_file} DIRECTORY)
  file(MAKE_DIRECTORY ${output_dir})
  execute_process(
    COMMAND ${cmd}
    OUTPUT_FILE ${output_file}
    RESULT_VARIABLE result
    ERROR_VARIABLE error
  )
  if(NOT result EQUAL 0)
    message(WARNING "libc size checks: failed to dump the predefined macros "
      "(${error}); target specific rules will be inactive.")
    file(WRITE ${output_file} "")
  endif()
endfunction()

# Follows entrypoint aliases (e.g. libc.src.math.logf ->
# libc.src.math.generic.logf) to the target that builds the object.
function(_libc_size_check_resolve_entrypoint output_var target)
  set(${output_var} "" PARENT_SCOPE)
  set(current ${target})
  while(TRUE)
    if(NOT TARGET ${current})
      return()
    endif()
    get_target_property(skipped ${current} "SKIPPED")
    if(skipped)
      return()
    endif()
    get_target_property(is_alias ${current} "IS_ALIAS")
    if(NOT is_alias)
      break()
    endif()
    get_target_property(current ${current} "DEPS")
  endwhile()
  set(${output_var} ${current} PARENT_SCOPE)
endfunction()

# Adds a check of the entrypoint objects whose target names match
# `ENTRYPOINT_REGEX` against the rules in `RULES`.
#
# Usage:
#   add_libc_object_size_check(
#     <name>
#     ENTRYPOINT_REGEX <regex matching fully qualified entrypoint targets>
#     RULES <rules JSON file>
#   )
#
# This creates the target `<name>` that writes `<name>.json` and `<name>.md`
# reports to the current binary directory and fails if a rule is violated.
function(add_libc_object_size_check name)
  cmake_parse_arguments(
    "SIZE_CHECK"
    "" # Optional arguments
    "ENTRYPOINT_REGEX;RULES" # Single value arguments
    "" # Multi value arguments
    ${ARGN}
  )
  set(manifest_lines "")
  set(object_targets "")
  set(macro_files "")
  foreach(entrypoint IN LISTS TARGET_LLVMLIBC_ENTRYPOINTS)
    if(NOT entrypoint MATCHES "${SIZE_CHECK_ENTRYPOINT_REGEX}")
      continue()
    endif()
    _libc_size_check_resolve_entrypoint(object_target ${entrypoint})
    if(NOT object_target)
      continue()
    endif()
    get_target_property(entrypoint_name ${object_target} "ENTRYPOINT_NAME")
    get_target_property(flags ${object_target} "FLAGS")
    if(NOT flags)
      set(flags "")
    endif()
    # Entrypoints can be compiled with different target options depending on
    # their flags (e.g. -mfma for FMA_OPT on x86-64), so dump the macros once
    # per distinct set of compile options.
    _get_common_compile_options(compile_options "${flags}")
    string(MD5 options_hash "${compile_options}")
    string(SUBSTRING ${options_hash} 0 12 options_hash)
    set(macro_file ${CMAKE_CURRENT_BINARY_DIR}/macros/${options_hash}.txt)
    if(NOT macro_file IN_LIST macro_files)
      _libc_size_check_dump_macros(${macro_file} ${compile_options})
      list(APPEND macro_files ${macro_file})
    endif()
    string(APPEND manifest_lines
      "${entrypoint_name}\t${macro_file}\t$<TARGET_OBJECTS:${object_target}>\n")
    list(APPEND object_targets ${object_target})
  endforeach()

  set(manifest ${CMAKE_CURRENT_BINARY_DIR}/${name}.manifest)
  file(GENERATE OUTPUT ${manifest} CONTENT "${manifest_lines}")
  add_custom_target(
    ${name}
    COMMAND
      ${Python3_EXECUTABLE} ${LIBC_SIZE_CHECK_SCRIPT} objects
      --manifest ${manifest}
      --rules ${SIZE_CHECK_RULES}
      "--math-optimizations=${LIBC_CONF_MATH_OPTIMIZATIONS}"
      --nm ${LIBC_SIZE_CHECK_NM}
      --size ${LIBC_SIZE_CHECK_SIZE}
      --json ${CMAKE_CURRENT_BINARY_DIR}/${name}.json
      --markdown ${CMAKE_CURRENT_BINARY_DIR}/${name}.md
    DEPENDS ${manifest} ${SIZE_CHECK_RULES} ${LIBC_SIZE_CHECK_SCRIPT}
    COMMENT "Checking the binary size of libc objects against ${SIZE_CHECK_RULES}"
    USES_TERMINAL
    VERBATIM
  )
  if(object_targets)
    add_dependencies(${name} ${object_targets})
  endif()
endfunction()

# Expands "SHELL:" prefixed options into separate arguments.
function(_libc_size_check_expand_options output_var)
  set(result "")
  foreach(opt IN LISTS ARGN)
    if(opt MATCHES "^SHELL:(.*)$")
      separate_arguments(shell_opts NATIVE_COMMAND "${CMAKE_MATCH_1}")
      list(APPEND result ${shell_opts})
    else()
      list(APPEND result ${opt})
    endif()
  endforeach()
  set(${output_var} ${result} PARENT_SCOPE)
endfunction()

# Adds a report of the size of minimal programs that call the functions in
# `PROBES` and are linked against the archives in `ARCHIVE_TARGETS`. The
# report is informational: the target only fails if the probes cannot be
# built at all. Probes that need the compiler runtime builtins are skipped
# when no builtins library is available.
#
# Usage:
#   add_libc_size_probes(
#     <name>
#     PROBES <probe list file>
#     MANIFEST <object manifest of add_libc_object_size_check>
#     ARCHIVE_TARGETS <libc archive targets>
#   )
function(add_libc_size_probes name)
  cmake_parse_arguments(
    "SIZE_PROBES"
    "" # Optional arguments
    "PROBES;MANIFEST" # Single value arguments
    "ARCHIVE_TARGETS" # Multi value arguments
    ${ARGN}
  )

  set(flags "")
  if(CMAKE_C_COMPILER_TARGET)
    list(APPEND flags "--target=${CMAKE_C_COMPILER_TARGET}")
  endif()
  separate_arguments(c_flags NATIVE_COMMAND "${CMAKE_C_FLAGS}")
  _libc_size_check_expand_options(default_options ${LIBC_COMPILE_OPTIONS_DEFAULT})
  list(APPEND flags ${c_flags} ${default_options})
  _libc_size_check_expand_options(link_flags ${LIBC_LINK_OPTIONS_DEFAULT})
  if(LLVM_USE_LINKER)
    list(APPEND link_flags "-fuse-ld=${LLVM_USE_LINKER}")
  endif()

  # Use the same compiler runtime builtins as the hermetic tests.
  set(builtins "")
  set(builtins_target "")
  if(LIBC_TEST_BUILTINS_LIBRARY)
    set(builtins ${LIBC_TEST_BUILTINS_LIBRARY})
  elseif(LIBC_TEST_BUILTINS_TARGET)
    set(builtins "$<TARGET_FILE:${LIBC_TEST_BUILTINS_TARGET}>")
    set(builtins_target ${LIBC_TEST_BUILTINS_TARGET})
  else()
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} ${flags} ${link_flags} -print-libgcc-file-name
      OUTPUT_VARIABLE libgcc_file
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE result
      ERROR_QUIET
    )
    if(result EQUAL 0 AND EXISTS "${libgcc_file}")
      set(builtins ${libgcc_file})
    else()
      message(STATUS "libc size probes: no compiler runtime builtins library "
        "found; probes that need it will be skipped. Set "
        "LIBC_TEST_BUILTINS_LIBRARY or LIBC_TEST_BUILTINS_TARGET to link them.")
    endif()
  endif()

  set(probe_args "")
  foreach(flag IN LISTS flags)
    list(APPEND probe_args "--cflag=${flag}")
  endforeach()
  foreach(flag IN LISTS link_flags)
    list(APPEND probe_args "--ldflag=${flag}")
  endforeach()
  foreach(archive IN LISTS SIZE_PROBES_ARCHIVE_TARGETS)
    list(APPEND probe_args "--archive=$<TARGET_FILE:${archive}>")
  endforeach()

  add_custom_target(
    ${name}
    COMMAND
      ${Python3_EXECUTABLE} ${LIBC_SIZE_CHECK_SCRIPT} probes
      --probes ${SIZE_PROBES_PROBES}
      --manifest ${SIZE_PROBES_MANIFEST}
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
      --cc ${CMAKE_C_COMPILER}
      ${probe_args}
      "--builtins=${builtins}"
      --nm ${LIBC_SIZE_CHECK_NM}
      --readelf ${LIBC_SIZE_CHECK_READELF}
      --json ${CMAKE_CURRENT_BINARY_DIR}/${name}.json
      --markdown ${CMAKE_CURRENT_BINARY_DIR}/${name}.md
    DEPENDS ${SIZE_PROBES_PROBES} ${SIZE_PROBES_MANIFEST} ${LIBC_SIZE_CHECK_SCRIPT}
    COMMENT "Linking libc size probes"
    USES_TERMINAL
    VERBATIM
  )
  add_dependencies(${name} ${SIZE_PROBES_ARCHIVE_TARGETS} ${builtins_target})
endfunction()
