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
  list(APPEND cmd ${cxx_flags})
  foreach(opt IN LISTS ARGN)
    if(opt MATCHES "^SHELL:(.*)$")
      separate_arguments(shell_opts NATIVE_COMMAND "${CMAKE_MATCH_1}")
      list(APPEND cmd ${shell_opts})
    else()
      list(APPEND cmd ${opt})
    endif()
  endforeach()
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
