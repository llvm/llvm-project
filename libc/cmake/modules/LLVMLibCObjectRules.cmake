set(OBJECT_LIBRARY_TARGET_TYPE "OBJECT_LIBRARY")
set(ENTRYPOINT_OBJ_TARGET_TYPE "ENTRYPOINT_OBJ")
set(ENTRYPOINT_EXT_TARGET_TYPE "ENTRYPOINT_EXT")

# Rule to check if a list of dependencies contains any entrypoint objects. Returns a list in entrypoint_deps.
function(check_entrypoint_deps entrypoint_deps)
  set(PUBLIC_DEPS "")
  set(fq_deps_list "")
  list(APPEND fq_deps_list ${ARGN})

  #don't warn for deps that are allowed, such as errno
  set(ALLOWED_DEPS
    "libc.src.errno.errno"
    "libc.src.setjmp.longjmp"
  )
  list(REMOVE_ITEM fq_deps_list ${ALLOWED_DEPS})

  foreach(dep IN LISTS fq_deps_list)
    if(NOT TARGET ${dep})
      continue()
    endif()

    get_target_property(target_type ${dep} "TARGET_TYPE")
    if(${target_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      list(APPEND PUBLIC_DEPS ${dep})
    endif()
  endforeach()
  set(${entrypoint_deps} ${PUBLIC_DEPS} PARENT_SCOPE)
endfunction()


# Rule which is essentially a wrapper over add_library to compile a set of
# sources to object files.
# Usage:
#     add_object_library(
#       <target_name>
#       HDRS <list of header files>
#       SRCS <list of source files>
#       [ALIAS] <If this object library is an alias for another object library.>
#       DEPENDS <list of dependencies; Should be a single item for ALIAS libraries>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       FLAGS <optional list of flags>
function(create_object_library fq_target_name)
  cmake_parse_arguments(
    "ADD_OBJECT"
    "ALIAS;NO_GPU_BUNDLE" # optional arguments
    "CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;COMPILE_OPTIONS;DEPENDS;FLAGS" # Multivalue arguments
    ${ARGN}
  )

  get_fq_deps_list(fq_deps_list ${ADD_OBJECT_DEPENDS})

  if(ADD_OBJECT_ALIAS)
    if(ADD_OBJECT_SRCS OR ADD_OBJECT_HDRS)
      message(FATAL_ERROR
              "${fq_target_name}: object library alias cannot have SRCS and/or HDRS.")
    endif()
    list(LENGTH fq_deps_list depends_size)
    if(NOT ${depends_size} EQUAL 1)
      message(FATAL_ERROR
              "${fq_targe_name}: object library alias should have exactly one DEPENDS.")
    endif()
    add_library(
      ${fq_target_name}
      ALIAS
      ${fq_deps_list}
    )
    return()
  endif()

  if(NOT ADD_OBJECT_SRCS)
    message(FATAL_ERROR "'add_object_library' rule requires SRCS to be specified.")
  endif()

  if(NOT ADD_OBJECT_CXX_STANDARD)
    set(ADD_OBJECT_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  set(internal_target_name ${fq_target_name}.__internal__)
  set(public_packaging_for_internal "-DLIBC_COPT_PUBLIC_PACKAGING")

  _get_common_compile_options(compile_options "${ADD_OBJECT_FLAGS}")
  list(APPEND compile_options ${ADD_OBJECT_COMPILE_OPTIONS})

  add_library(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_OBJECT_SRCS}
    ${ADD_OBJECT_HDRS}
  )
  target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  target_compile_options(${fq_target_name} PRIVATE ${compile_options})

  #loop through the deps, check if any have the TARGET_TYPE of ENTRYPOINT_OBJ_TARGET_TYPE, and print a warning if they do.
  if(LIBC_CMAKE_VERBOSE_LOGGING)
    set(entrypoint_deps "")
    if(NOT "${fq_deps_list}" STREQUAL "")
      check_entrypoint_deps(entrypoint_deps ${fq_deps_list})
    endif()
    if(NOT "${entrypoint_deps}" STREQUAL "")
      message(WARNING "Object ${fq_target_name} depends on public entrypoint(s) ${entrypoint_deps}.
      Depending on public entrypoints is not allowed in internal code.")
    endif()
  endif()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding object library ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  list(APPEND fq_deps_list libc.src.__support.macros.config)
  list(REMOVE_DUPLICATES fq_deps_list)
  add_dependencies(${fq_target_name} ${fq_deps_list})
  # Add deps as link libraries to inherit interface compile and link options.
  target_link_libraries(${fq_target_name} PUBLIC ${fq_deps_list})

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      TARGET_TYPE ${OBJECT_LIBRARY_TARGET_TYPE}
      CXX_STANDARD ${ADD_OBJECT_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_OBJECT_FLAGS}"
  )

  # If we built a separate internal target we want to use those target objects
  # for testing instead of the exported target.
  set(target_objects ${fq_target_name})
  if(TARGET ${internal_target_name})
    set(target_objects ${internal_target_name})
  endif()

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      OBJECT_FILES "$<TARGET_OBJECTS:${target_objects}>"
  )
endfunction(create_object_library)

function(add_object_library target_name)
  add_target_with_flags(
    ${target_name}
    CREATE_TARGET create_object_library
    ${ARGN})
endfunction(add_object_library)


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

  set(entrypoint_target_type ${ENTRYPOINT_OBJ_TARGET_TYPE})
  list(FIND TARGET_ENTRYPOINT_NAME_LIST ${ADD_ENTRYPOINT_OBJ_NAME} entrypoint_name_index)
  if(${entrypoint_name_index} EQUAL -1)
    add_custom_target(${fq_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        "ENTRYPOINT_NAME" ${ADD_ENTRYPOINT_OBJ_NAME}
        "TARGET_TYPE" ${entrypoint_target_type}
        "OBJECT_FILE" ""
        "OBJECT_FILE_RAW" ""
        "DEPS" ""
        "SKIPPED" "YES"
    )
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      message(STATUS "Skipping libc entrypoint ${fq_target_name}.")
    endif()
    return()
  endif()

  set(internal_target_name ${fq_target_name}.__internal__)

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
    if((NOT obj_type) OR (NOT ${obj_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}))
      message(FATAL_ERROR "The aliasee of an entrypoint alias should be an entrypoint.")
    endif()

    get_target_property(object_file ${fq_dep_name} "OBJECT_FILE")
    get_target_property(object_file_raw ${fq_dep_name} "OBJECT_FILE_RAW")

    # If the system cannot build the GPU tests we simply make a dummy target.
    if(LIBC_TARGET_OS_IS_GPU AND LIBC_GPU_TESTS_DISABLED)
      add_custom_target(${internal_target_name})
    else()
      add_library(
        ${internal_target_name}
        EXCLUDE_FROM_ALL
        OBJECT
        ${object_file_raw}
      )
    endif()

    add_dependencies(${internal_target_name} ${fq_dep_name})
    add_library(
      ${fq_target_name}
      EXCLUDE_FROM_ALL
      OBJECT
      ${object_file}
    )
    add_dependencies(${fq_target_name} ${fq_dep_name} ${internal_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
        TARGET_TYPE ${entrypoint_target_type}
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
  if(NOT ADD_ENTRYPOINT_OBJ_CXX_STANDARD)
    set(ADD_ENTRYPOINT_OBJ_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  _get_common_compile_options(common_compile_options "${ADD_ENTRYPOINT_OBJ_FLAGS}")
  list(APPEND common_compile_options ${ADD_ENTRYPOINT_OBJ_COMPILE_OPTIONS})
  get_fq_deps_list(fq_deps_list ${ADD_ENTRYPOINT_OBJ_DEPENDS})

  #loop through the deps, check if any have the TARGET_TYPE of entrypoint_target_type, and print a warning if they do.
  if(LIBC_CMAKE_VERBOSE_LOGGING)
    set(entrypoint_deps "")
    if(NOT "${fq_deps_list}" STREQUAL "")
      check_entrypoint_deps(entrypoint_deps ${fq_deps_list})
    endif()
    if(NOT "${entrypoint_deps}" STREQUAL "")
      message(WARNING "Entrypoint ${fq_target_name} depends on public entrypoint(s) ${entrypoint_deps}.
      Depending on public entrypoints is not allowed in internal code.")
    endif()
  endif()

  set(full_deps_list ${fq_deps_list} libc.src.__support.common)

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding entrypoint object ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_ENTRYPOINT_OBJ_DEPENDS)
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
  target_include_directories(${internal_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${internal_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  add_dependencies(${internal_target_name} ${full_deps_list})
  target_link_libraries(${internal_target_name} ${full_deps_list})

  add_library(
    ${fq_target_name}
    # We want an object library as the objects will eventually get packaged into
    # an archive (like libc.a).
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_ENTRYPOINT_OBJ_SRCS}
    ${ADD_ENTRYPOINT_OBJ_HDRS}
  )
  target_compile_options(${fq_target_name} BEFORE PRIVATE ${common_compile_options} -DLIBC_COPT_PUBLIC_PACKAGING)
  target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  add_dependencies(${fq_target_name} ${full_deps_list})
  target_link_libraries(${fq_target_name} ${full_deps_list})

  # Builtin recognition causes issues when trying to implement the builtin
  # functions themselves. The GPU backends do not use libcalls so we disable the
  # known problematic ones on the entrypoints that implement them.
  if(LIBC_TARGET_OS_IS_GPU)
    set(libc_builtins bcmp strlen memmem bzero memcmp memcpy memmem memmove
                      memset strcmp strstr)
    if(${ADD_ENTRYPOINT_OBJ_NAME} IN_LIST libc_builtins)
      target_compile_options(${fq_target_name} PRIVATE -fno-builtin-${ADD_ENTRYPOINT_OBJ_NAME})
    endif()
  endif()

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
      TARGET_TYPE ${entrypoint_target_type}
      OBJECT_FILE "$<TARGET_OBJECTS:${fq_target_name}>"
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
  )

  if(TARGET ${internal_target_name})
    set_target_properties(
      ${internal_target_name}
      PROPERTIES
        CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
        FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        # TODO: We don't need to list internal object files if the internal
        # target is a normal static library.
        OBJECT_FILE_RAW "$<TARGET_OBJECTS:${internal_target_name}>"
    )
  endif()

  if(LLVM_LIBC_ENABLE_LINTING AND TARGET ${internal_target_name})
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

    add_custom_target(
      ${fq_target_name}.__lint__
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
      COMMENT "Linting... ${fq_target_name}"
      DEPENDS ${internal_target_name} ${ADD_ENTRYPOINT_OBJ_SRCS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(libc-lint ${fq_target_name}.__lint__)
  endif()

endfunction(create_entrypoint_object)

function(add_entrypoint_object target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "" # Optional arguments
    "NAME" # Single value arguments
    "" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_ENTRYPOINT_OBJ_NAME)
    set(ADD_ENTRYPOINT_OBJ_NAME ${target_name})
  endif()

  add_target_with_flags(
    ${target_name}
    NAME ${ADD_ENTRYPOINT_OBJ_NAME}
    CREATE_TARGET create_entrypoint_object
    ${ADD_ENTRYPOINT_OBJ_UNPARSED_ARGUMENTS}
  )
endfunction(add_entrypoint_object)

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

# Helper to define a function with multiple implementations
# - Computes flags to satisfy required/rejected features and arch,
# - Declares an entry point,
# - Attach the REQUIRE_CPU_FEATURES property to the target,
# - Add the fully qualified target to `${name}_implementations` global property for tests.
function(add_implementation name impl_name)
  cmake_parse_arguments(
    "ADD_IMPL"
    "" # Optional arguments
    "" # Single value arguments
    "REQUIRE;SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;MLLVM_COMPILE_OPTIONS" # Multi value arguments
    ${ARGN})

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Note that '-mllvm' needs to be prefixed with 'SHELL:' to prevent CMake flag deduplication.
    foreach(opt IN LISTS ADD_IMPL_MLLVM_COMPILE_OPTIONS)
      list(APPEND ADD_IMPL_COMPILE_OPTIONS "SHELL:-mllvm ${opt}")
    endforeach()
  endif()

  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # Prevent warning when passing x86 SIMD types as template arguments.
    # e.g. "warning: ignoring attributes on template argument ‘__m128i’ [-Wignored-attributes]"
    list(APPEND ADD_IMPL_COMPILE_OPTIONS "-Wno-ignored-attributes")
  endif()

  add_entrypoint_object(${impl_name}
    NAME ${name}
    SRCS ${ADD_IMPL_SRCS}
    HDRS ${ADD_IMPL_HDRS}
    DEPENDS ${ADD_IMPL_DEPENDS}
    COMPILE_OPTIONS ${libc_opt_high_flag} ${ADD_IMPL_COMPILE_OPTIONS}
  )
  get_fq_target_name(${impl_name} fq_target_name)
  set_target_properties(${fq_target_name} PROPERTIES REQUIRE_CPU_FEATURES "${ADD_IMPL_REQUIRE}")
  set_property(GLOBAL APPEND PROPERTY "${name}_implementations" "${fq_target_name}")
endfunction()
