set(OBJECT_LIBRARY_TARGET_TYPE "OBJECT_LIBRARY")

function(_get_common_compile_options output_var)
  set(compile_options ${LIBC_COMPILE_OPTIONS_DEFAULT} ${ARGN})
  if(NOT ${LIBC_TARGET_OS} STREQUAL "windows")
    set(compile_options ${compile_options} -fpie -ffreestanding -fno-builtin)
  endif()
  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    list(APPEND compile_options "-fno-exceptions")
    list(APPEND compile_options "-fno-unwind-tables")
    list(APPEND compile_options "-fno-asynchronous-unwind-tables")
    list(APPEND compile_options "-fno-rtti")
  elseif(MSVC)
    list(APPEND compile_options "/EHs-c-")
    list(APPEND compile_options "/GR-")
  endif()
  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction()

# Rule which is essentially a wrapper over add_library to compile a set of
# sources to object files.
# Usage:
#     add_object_library(
#       <target_name>
#       HDRS <list of header files>
#       SRCS <list of source files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       FLAGS <optional list of flags>
function(create_object_library fq_target_name)
  cmake_parse_arguments(
    "ADD_OBJECT"
    "" # No optional arguments
    "CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;COMPILE_OPTIONS;DEPENDS;FLAGS" # Multivalue arguments
    ${ARGN}
  )

  if(NOT ADD_OBJECT_SRCS)
    message(FATAL_ERROR "'add_object_library' rule requires SRCS to be specified.")
  endif()

  add_library(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_OBJECT_SRCS}
    ${ADD_OBJECT_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_BUILD_DIR}/include
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
  )
  _get_common_compile_options(compile_options ${ADD_OBJECT_COMPILE_OPTIONS})
  target_compile_options(${fq_target_name} PRIVATE ${compile_options})

  get_fq_deps_list(fq_deps_list ${ADD_OBJECT_DEPENDS})

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding object library ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  if(fq_deps_list)
    add_dependencies(${fq_target_name} ${fq_deps_list})
  endif()

  if(NOT ADD_OBJECT_CXX_STANDARD)
    set(ADD_OBJECT_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      TARGET_TYPE ${OBJECT_LIBRARY_TARGET_TYPE}
      OBJECT_FILES "$<TARGET_OBJECTS:${fq_target_name}>"
      CXX_STANDARD ${ADD_OBJECT_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_OBJECT_FLAGS}"
  )
endfunction(create_object_library)

# Internal function, used by `add_object_library`.
function(expand_flags_for_object_library target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "IGNORE_MARKER" # Optional arguments
    "" # Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    create_object_library(
      ${target_name}
      DEPENDS ${EXPAND_FLAGS_DEPENDS}
      FLAGS ${EXPAND_FLAGS_FLAGS}
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
    )
    return()
  endif()

  list(POP_FRONT flags flag)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_object_library(
      ${target_name}
      "${flags}"
      DEPENDS "${EXPAND_FLAGS_DEPENDS}" IGNORE_MARKER
      FLAGS "${EXPAND_FLAGS_FLAGS}" IGNORE_MARKER
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if(NOT "${modifier}")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_object_library(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS "${NEW_DEPS}" IGNORE_MARKER
    FLAGS "${NEW_FLAGS}" IGNORE_MARKER
    "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
  )
endfunction(expand_flags_for_object_library)

function(add_object_library target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "" # Single value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})

  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Object library ${fq_target_name} has FLAGS: ${flags}")
  endif()

  expand_flags_for_object_library(
    ${fq_target_name}
    "${flags}"
    DEPENDS "${fq_deps_list}" IGNORE_MARKER
    FLAGS "${flags}" IGNORE_MARKER
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_object_library)

set(ENTRYPOINT_OBJ_TARGET_TYPE "ENTRYPOINT_OBJ")

# A rule for entrypoint object targets.
# Usage:
#     add_entrypoint_object(
#       <target_name>
#       [ALIAS|REDIRECTED] # Specified if the entrypoint is redirected or an alias.
#       [NAME] <the C name of the entrypoint if different from target_name>
#       SRCS <list of .cpp files>
#       HDRS <list of .h files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       SPECIAL_OBJECTS <optional list of special object targets added by the rule `add_object`>
#       FLAGS <optional list of flags>
#     )
function(create_entrypoint_object fq_target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "ALIAS;REDIRECTED" # Optional argument
    "NAME;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;FLAGS"  # Multi value arguments
    ${ARGN}
  )

  list(FIND TARGET_ENTRYPOINT_NAME_LIST ${ADD_ENTRYPOINT_OBJ_NAME} entrypoint_name_index)
  if(${entrypoint_name_index} EQUAL -1)
    add_custom_target(${fq_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        "ENTRYPOINT_NAME" ${ADD_ENTRYPOINT_OBJ_NAME}
        "TARGET_TYPE" ${ENTRYPOINT_OBJ_TARGET_TYPE}
        "OBJECT_FILE" ""
        "OBJECT_FILE_RAW" ""
        "DEPS" ""
        "SKIPPED" "YES"
    )
    message(STATUS "Skipping libc entrypoint ${fq_target_name}.")
    return()
  endif()

  if(ADD_ENTRYPOINT_OBJ_ALIAS)
    # Alias targets help one add aliases to other entrypoint object targets.
    # One can use alias targets setup OS/machine independent entrypoint targets.
    list(LENGTH ADD_ENTRYPOINT_OBJ_DEPENDS deps_size)
    if(NOT (${deps_size} EQUAL "1"))
      message(FATAL_ERROR "An entrypoint alias should have exactly one dependency.")
    endif()
    list(GET ADD_ENTRYPOINT_OBJ_DEPENDS 0 dep_target)
    get_fq_dep_name(fq_dep_name ${dep_target})

    if(SHOW_INTERMEDIATE_OBJECTS)
      message(STATUS "Adding entrypoint object ${fq_target_name} as an alias of"
              " ${fq_dep_name}")
    endif()

    if(NOT TARGET ${fq_dep_name})
      message(WARNING "Aliasee ${fq_dep_name} for entrypoint alias ${target_name} missing; "
                      "Target ${target_name} will be ignored.")
      return()
    endif()

    get_target_property(obj_type ${fq_dep_name} "TARGET_TYPE")
    if((NOT obj_type) OR (NOT (${obj_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})))
      message(FATAL_ERROR "The aliasee of an entrypoint alias should be an entrypoint.")
    endif()

    add_custom_target(${fq_target_name})
    add_dependencies(${fq_target_name} ${fq_dep_name})
    get_target_property(object_file ${fq_dep_name} "OBJECT_FILE")
    get_target_property(object_file_raw ${fq_dep_name} "OBJECT_FILE_RAW")
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
        TARGET_TYPE ${ENTRYPOINT_OBJ_TARGET_TYPE}
        IS_ALIAS "YES"
        OBJECT_FILE ""
        OBJECT_FILE_RAW ""
        DEPS "${fq_dep_name}"
        FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
    return()
  endif()

  if(NOT ADD_ENTRYPOINT_OBJ_SRCS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires SRCS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_HDRS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires HDRS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_CXX_STANDARD)
    set(ADD_ENTRYPOINT_OBJ_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  _get_common_compile_options(common_compile_options ${ADD_ENTRYPOINT_OBJ_COMPILE_OPTIONS})
  set(internal_target_name ${fq_target_name}.__internal__)
  set(include_dirs ${LIBC_BUILD_DIR}/include ${LIBC_SOURCE_DIR} ${LIBC_BUILD_DIR})
  get_fq_deps_list(fq_deps_list ${ADD_ENTRYPOINT_OBJ_DEPENDS})
  set(full_deps_list ${fq_deps_list} libc.src.__support.common)

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding entrypoint object ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  add_library(
    ${internal_target_name}
    # TODO: We don't need an object library for internal consumption.
    # A future change should switch this to a normal static library.
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_ENTRYPOINT_OBJ_SRCS}
    ${ADD_ENTRYPOINT_OBJ_HDRS}
  )
  target_compile_options(${internal_target_name} BEFORE PRIVATE ${common_compile_options})
  target_include_directories(${internal_target_name} PRIVATE ${include_dirs})
  add_dependencies(${internal_target_name} ${full_deps_list})
  set_target_properties(
    ${internal_target_name}
    PROPERTIES
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
  )

  add_library(
    ${fq_target_name}
    # We want an object library as the objects will eventually get packaged into
    # an archive (like libc.a).
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_ENTRYPOINT_OBJ_SRCS}
    ${ADD_ENTRYPOINT_OBJ_HDRS}
  )
  target_compile_options(${fq_target_name} BEFORE PRIVATE ${common_compile_options} -DLLVM_LIBC_PUBLIC_PACKAGING)
  target_include_directories(${fq_target_name} PRIVATE ${include_dirs})
  add_dependencies(${fq_target_name} ${full_deps_list})

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
      TARGET_TYPE ${ENTRYPOINT_OBJ_TARGET_TYPE}
      OBJECT_FILE "$<TARGET_OBJECTS:${fq_target_name}>"
      # TODO: We don't need to list internal object files if the internal
      # target is a normal static library.
      OBJECT_FILE_RAW "$<TARGET_OBJECTS:${internal_target_name}>"
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
  )

  if(LLVM_LIBC_ENABLE_LINTING)
    if(NOT LLVM_LIBC_CLANG_TIDY)
      message(FATAL_ERROR "Something is wrong!  LLVM_LIBC_ENABLE_LINTING is "
              "ON but LLVM_LIBC_CLANG_TIDY is not set.")
    endif()

    # We only want a second invocation of clang-tidy to run
    # restrict-system-libc-headers if the compiler-resource-dir was set in
    # order to prevent false-positives due to a mismatch between the host
    # compiler and the compiled clang-tidy.
    if(COMPILER_RESOURCE_DIR)
      # We run restrict-system-libc-headers with --system-headers to prevent
      # transitive inclusion through compler provided headers.
      set(restrict_system_headers_check_invocation
        COMMAND ${LLVM_LIBC_CLANG_TIDY} --system-headers
        --checks="-*,llvmlibc-restrict-system-libc-headers"
        # We explicitly set the resource dir here to match the
        # resource dir of the host compiler.
        "--extra-arg=-resource-dir=${COMPILER_RESOURCE_DIR}"
        --quiet
        -p ${PROJECT_BINARY_DIR}
        ${ADD_ENTRYPOINT_OBJ_SRCS}
      )
    else()
      set(restrict_system_headers_check_invocation
        COMMAND ${CMAKE_COMMAND} -E echo "Header file check skipped")
    endif()

    set(lint_timestamp "${CMAKE_CURRENT_BINARY_DIR}/.${target_name}.__lint_timestamp__")
    add_custom_command(
      OUTPUT ${lint_timestamp}
      # --quiet is used to surpress warning statistics from clang-tidy like:
      #     Suppressed X warnings (X in non-user code).
      # There seems to be a bug in clang-tidy where by even with --quiet some
      # messages from clang's own diagnostics engine leak through:
      #     X warnings generated.
      # Until this is fixed upstream, we use -fno-caret-diagnostics to surpress
      # these.
      COMMAND ${LLVM_LIBC_CLANG_TIDY}
              "--extra-arg=-fno-caret-diagnostics" --quiet
              # Path to directory containing compile_commands.json
              -p ${PROJECT_BINARY_DIR}
              ${ADD_ENTRYPOINT_OBJ_SRCS}
      # See above: this might be a second invocation of clang-tidy depending on
      # the conditions above.
      ${restrict_system_headers_check_invocation}
      # We have two options for running commands, add_custom_command and
      # add_custom_target. We don't want to run the linter unless source files
      # have changed. add_custom_target explicitly runs everytime therefore we
      # use add_custom_command. This function requires an output file and since
      # linting doesn't produce a file, we create a dummy file using a
      # crossplatform touch.
      COMMAND "${CMAKE_COMMAND}" -E touch ${lint_timestamp}
      COMMENT "Linting... ${target_name}"
      DEPENDS clang-tidy ${internal_target_name} ${ADD_ENTRYPOINT_OBJ_SRCS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()

endfunction(create_entrypoint_object)

# Internal function, used by `add_entrypoint_object`.
function(expand_flags_for_entrypoint_object target_name flags)
  cmake_parse_arguments(
    "EXPAND_FLAGS"
    "IGNORE_MARKER" # Optional arguments
    "" # Single-value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  list(LENGTH flags nflags)
  if(NOT ${nflags})
    create_entrypoint_object(
      ${target_name}
      DEPENDS ${EXPAND_FLAGS_DEPENDS}
      FLAGS ${EXPAND_FLAGS_FLAGS}
      ${EXPAND_FLAGS_UNPARSED_ARGUMENTS}
    )
    return()
  endif()

  list(POP_FRONT flags flag)
  extract_flag_modifier(${flag} real_flag modifier)

  if(NOT "${modifier}" STREQUAL "NO")
    expand_flags_for_entrypoint_object(
      ${target_name}
      "${flags}"
      DEPENDS "${EXPAND_FLAGS_DEPENDS}" IGNORE_MARKER
      FLAGS "${EXPAND_FLAGS_FLAGS}" IGNORE_MARKER
      "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
    )
  endif()

  if("${real_flag}" STREQUAL "" OR "${modifier}" STREQUAL "ONLY")
    return()
  endif()

  set(NEW_FLAGS ${EXPAND_FLAGS_FLAGS})
  list(REMOVE_ITEM NEW_FLAGS ${flag})
  get_fq_dep_list_without_flag(NEW_DEPS ${real_flag} ${EXPAND_FLAGS_DEPENDS})

  # Only target with `flag` has `.__NO_flag` target, `flag__NO` and
  # `flag__ONLY` do not.
  if(NOT "${modifier}")
    set(TARGET_NAME "${target_name}.__NO_${flag}")
  else()
    set(TARGET_NAME "${target_name}")
  endif()

  expand_flags_for_entrypoint_object(
    ${TARGET_NAME}
    "${flags}"
    DEPENDS "${NEW_DEPS}" IGNORE_MARKER
    FLAGS "${NEW_FLAGS}" IGNORE_MARKER
    "${EXPAND_FLAGS_UNPARSED_ARGUMENTS}"
  )
endfunction(expand_flags_for_entrypoint_object)

function(add_entrypoint_object target_name)
  cmake_parse_arguments(
    "ADD_TO_EXPAND"
    "" # Optional arguments
    "NAME" # Single value arguments
    "DEPENDS;FLAGS" # Multi-value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_TO_EXPAND_DEPENDS AND ("${SHOW_INTERMEDIATE_OBJECTS}" STREQUAL "DEPS"))
    message(STATUS "Gathering FLAGS from dependencies for ${fq_target_name}")
  endif()

  get_fq_deps_list(fq_deps_list ${ADD_TO_EXPAND_DEPENDS})
  get_flags_from_dep_list(deps_flag_list ${fq_deps_list})

  list(APPEND ADD_TO_EXPAND_FLAGS ${deps_flag_list})
  remove_duplicated_flags("${ADD_TO_EXPAND_FLAGS}" flags)
  list(SORT flags)

  if(SHOW_INTERMEDIATE_OBJECTS AND flags)
    message(STATUS "Object library ${fq_target_name} has FLAGS: ${flags}")
  endif()

  if(NOT ADD_TO_EXPAND_NAME)
    set(ADD_TO_EXPAND_NAME ${target_name})
  endif()

  expand_flags_for_entrypoint_object(
    ${fq_target_name}
    "${flags}"
    NAME ${ADD_TO_EXPAND_NAME} IGNORE_MARKER
    DEPENDS "${fq_deps_list}" IGNORE_MARKER
    FLAGS "${flags}" IGNORE_MARKER
    ${ADD_TO_EXPAND_UNPARSED_ARGUMENTS}
  )
endfunction(add_entrypoint_object)

set(ENTRYPOINT_EXT_TARGET_TYPE "ENTRYPOINT_EXT")

# A rule for external entrypoint targets.
# Usage:
#     add_entrypoint_external(
#       <target_name>
#       DEPENDS <list of dependencies>
#     )
function(add_entrypoint_external target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_EXT"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS"  # Multi value arguments
    ${ARGN}
  )
  get_fq_target_name(${target_name} fq_target_name)
  set(entrypoint_name ${target_name})

  add_custom_target(${fq_target_name})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "ENTRYPOINT_NAME" ${entrypoint_name}
      "TARGET_TYPE" ${ENTRYPOINT_EXT_TARGET_TYPE}
      "DEPS" "${ADD_ENTRYPOINT_EXT_DEPENDS}"
  )

endfunction(add_entrypoint_external)

# Rule build a redirector object file.
function(add_redirector_object target_name)
  cmake_parse_arguments(
    "REDIRECTOR_OBJECT"
    "" # No optional arguments
    "SRC" # The cpp file in which the redirector is defined.
    "" # No multivalue arguments
    ${ARGN}
  )
  if(NOT REDIRECTOR_OBJECT_SRC)
    message(FATAL_ERROR "'add_redirector_object' rule requires SRC option listing one source file.")
  endif()

  add_library(
    ${target_name}
    EXCLUDE_FROM_ALL
    OBJECT
    ${REDIRECTOR_OBJECT_SRC}
  )
  target_compile_options(
    ${target_name}
    BEFORE PRIVATE -fPIC ${LIBC_COMPILE_OPTIONS_DEFAULT}
  )
endfunction(add_redirector_object)
