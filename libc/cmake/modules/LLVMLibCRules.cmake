include(LLVMLibCTargetNameUtils)

# A rule for self contained header file targets.
# This rule merely copies the header file from the current source directory to
# the current binary directory.
# Usage:
#     add_header(
#       <target name>
#       HDR <header file>
#     )
function(add_header target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    ""    # No optional arguments
    "HDR" # Single value arguments
    "DEPENDS"
    ${ARGN}
  )
  if(NOT ADD_HEADER_HDR)
    message(FATAL_ERROR "'add_header' rules requires the HDR argument specifying a headef file.")
  endif()

  set(dest_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_HEADER_HDR})
  set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_HEADER_HDR})

  add_custom_command(
    OUTPUT ${dest_file}
    COMMAND cp ${src_file} ${dest_file}
    DEPENDS ${src_file}
  )

  get_fq_target_name(${target_name} fq_target_name)
  add_custom_target(
    ${fq_target_name}
    DEPENDS ${dest_file}
  )

  if(ADD_HEADER_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_HEADER_DEPENDS})
    add_dependencies(
      ${fq_target_name} ${fq_deps_list}
    )
  endif()
endfunction(add_header)

# A rule for generated header file targets.
# Usage:
#     add_gen_header(
#       <target name>
#       DEF_FILE <.h.def file>
#       GEN_HDR <generated header file name>
#       PARAMS <list of name=value pairs>
#       DATA_FILES <list input data files>
#     )
function(add_gen_header target_name)
  cmake_parse_arguments(
    "ADD_GEN_HDR"
    "" # No optional arguments
    "DEF_FILE;GEN_HDR" # Single value arguments
    "PARAMS;DATA_FILES;DEPENDS"     # Multi value arguments
    ${ARGN}
  )
  if(NOT ADD_GEN_HDR_DEF_FILE)
    message(FATAL_ERROR "`add_gen_hdr` rule requires DEF_FILE to be specified.")
  endif()
  if(NOT ADD_GEN_HDR_GEN_HDR)
    message(FATAL_ERROR "`add_gen_hdr` rule requires GEN_HDR to be specified.")
  endif()

  set(out_file ${CMAKE_CURRENT_BINARY_DIR}/${ADD_GEN_HDR_GEN_HDR})
  set(in_file ${CMAKE_CURRENT_SOURCE_DIR}/${ADD_GEN_HDR_DEF_FILE})

  set(fq_data_files "")
  if(ADD_GEN_HDR_DATA_FILES)
    foreach(data_file IN LISTS ADD_GEN_HDR_DATA_FILES)
      list(APPEND fq_data_files "${CMAKE_CURRENT_SOURCE_DIR}/${data_file}")
    endforeach(data_file)
  endif()

  set(replacement_params "")
  if(ADD_GEN_HDR_PARAMS)
    list(APPEND replacement_params "--args" ${ADD_GEN_HDR_PARAMS})
  endif()

  set(gen_hdr_script "${LIBC_BUILD_SCRIPTS_DIR}/gen_hdr.py")

  file(GLOB td_includes ${LIBC_SOURCE_DIR}/spec/*.td)

  add_custom_command(
    OUTPUT ${out_file}
    COMMAND $<TARGET_FILE:libc-hdrgen> -o ${out_file} --header ${ADD_GEN_HDR_GEN_HDR}
            --def ${in_file} ${replacement_params} -I ${LIBC_SOURCE_DIR}
            ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td

    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${in_file} ${fq_data_files} ${td_includes} 
            ${LIBC_SOURCE_DIR}/config/${LIBC_TARGET_OS}/api.td libc-hdrgen
  )

  get_fq_target_name(${target_name} fq_target_name)
  if(ADD_GEN_HDR_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_GEN_HDR_DEPENDS})
  endif()
  add_custom_target(
    ${fq_target_name}
    DEPENDS ${out_file} ${fq_deps_list}
  )
endfunction(add_gen_header)

set(OBJECT_LIBRARY_TARGET_TYPE "OBJECT_LIBRARY")

# Rule which is essentially a wrapper over add_library to compile a set of
# sources to object files.
# Usage:
#     add_object_library(
#       <target_name>
#       HDRS <list of header files>
#       SRCS <list of source files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
function(add_object_library target_name)
  cmake_parse_arguments(
    "ADD_OBJECT"
    "" # No option arguments
    "" # Single value arguments
    "SRCS;HDRS;COMPILE_OPTIONS;DEPENDS" # Multivalue arguments
    ${ARGN}
  )

  if(NOT ADD_OBJECT_SRCS)
    message(FATAL_ERROR "'add_object_library' rule requires SRCS to be specified.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  add_library(
    ${fq_target_name}
    OBJECT
    ${ADD_OBJECT_SRCS}
    ${ADD_OBJECT_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      "${LIBC_BUILD_DIR}/include;${LIBC_SOURCE_DIR};${LIBC_BUILD_DIR}"
  )
  if(ADD_OBJECT_COMPILE_OPTIONS)
    target_compile_options(
      ${fq_target_name}
      PRIVATE ${ADD_OBJECT_COMPILE_OPTIONS}
    )
  endif()

  set(all_object_files $<TARGET_OBJECTS:${fq_target_name}>)
  if(ADD_OBJECT_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_OBJECT_DEPENDS})
    add_dependencies(
      ${fq_target_name}
      ${fq_deps_list}
    )
    foreach(obj_target IN LISTS fq_deps_list)
      if(NOT TARGET obj_target)
        # Not all targets will be visible. So, we will ignore those which aren't
        # visible yet.
        continue()
      endif()
      get_target_property(obj_type ${obj_target} "TARGET_TYPE")
      if((NOT obj_type) OR (NOT (${obj_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})))
        continue()
      endif()
      # If a dependency is also a object file library, we will collect the list of
      # object files from it.
      get_target_property(obj_files ${obj_target} "OBJECT_FILES")
      list(APPEND all_object_files ${obj_files})
    endforeach(obj_target)
  endif()
  list(REMOVE_DUPLICATES all_object_files)

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "TARGET_TYPE" ${OBJECT_LIBRARY_TARGET_TYPE}
      "OBJECT_FILES" "${all_object_files}"
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
#     )
function(add_entrypoint_object target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "ALIAS;REDIRECTED" # Optional argument
    "NAME" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS"  # Multi value arguments
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)

  if(ADD_ENTRYPOINT_OBJ_ALIAS)
    # Alias targets help one add aliases to other entrypoint object targets.
    # One can use alias targets setup OS/machine independent entrypoint targets.
    list(LENGTH ADD_ENTRYPOINT_OBJ_DEPENDS deps_size)
    if(NOT (${deps_size} EQUAL "1"))
      message(FATAL_ERROR "An entrypoint alias should have exactly one dependency.")
    endif()
    list(GET ADD_ENTRYPOINT_OBJ_DEPENDS 0 dep_target)
    get_fq_dep_name(fq_dep_name ${dep_target})
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
    get_target_property(all_objects ${fq_dep_name} "OBJECT_FILES")
    get_target_property(all_objects_raw ${fq_dep_name} "OBJECT_FILES_RAW")
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        "TARGET_TYPE" ${ENTRYPOINT_OBJ_TARGET_TYPE}
        "OBJECT_FILES" "${all_objects}"
        "OBJECT_FILES_RAW" "${all_objects_raw}"
    )
    return()
  endif()

  if(NOT ADD_ENTRYPOINT_OBJ_SRCS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires SRCS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_HDRS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires HDRS to be specified.")
  endif()

  set(entrypoint_name ${target_name})
  if(ADD_ENTRYPOINT_OBJ_NAME)
    set(entrypoint_name ${ADD_ENTRYPOINT_OBJ_NAME})
  endif()

  set(objects_target_name "${fq_target_name}_objects")

  add_library(
    ${objects_target_name}
    # We want an object library as the objects will eventually get packaged into
    # an archive (like libc.a).
    OBJECT
    ${ADD_ENTRYPOINT_OBJ_SRCS}
    ${ADD_ENTRYPOINT_OBJ_HDRS}
  )
  target_compile_options(
    ${objects_target_name}
    BEFORE
    PRIVATE
      -fpie ${LLVM_CXX_STD_default}
  )
  target_include_directories(
    ${objects_target_name}
    PRIVATE
      "${LIBC_BUILD_DIR}/include;${LIBC_SOURCE_DIR};${LIBC_BUILD_DIR}"
  )
  add_dependencies(
    ${objects_target_name}
    libc.src.__support.common
  )
  set(dep_objects "")
  if(ADD_ENTRYPOINT_OBJ_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_ENTRYPOINT_OBJ_DEPENDS})
    add_dependencies(
      ${objects_target_name}
      ${fq_deps_list}
    )
    foreach(dep_target IN LISTS fq_deps_list)
      if(NOT TARGET ${dep_target})
        # Not all targets will be visible. So, we will ignore those which aren't
        # visible yet.
        continue()
      endif()
      get_target_property(obj_type ${dep_target} "TARGET_TYPE")
      if((NOT obj_type) OR (NOT (${obj_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})))
        # Even from among the visible targets, we will collect object files
        # only from add_object_library targets.
        continue()
      endif()
      # Calling get_target_property requires that the target be visible at this
      # point. For object library dependencies, this is a reasonable requirement.
      # We can revisit this in future if we need cases which break under this
      # requirement.
      get_target_property(obj_files ${dep_target} "OBJECT_FILES")
      list(APPEND dep_objects ${obj_files})
    endforeach(dep_target)
  endif()
  list(REMOVE_DUPLICATES dep_objects)

  if(ADD_ENTRYPOINT_OBJ_COMPILE_OPTIONS)
    target_compile_options(
      ${objects_target_name}
      PRIVATE ${ADD_ENTRYPOINT_OBJ_COMPILE_OPTIONS}
    )
  endif()

  set(object_file_raw "${CMAKE_CURRENT_BINARY_DIR}/${target_name}_raw.o")
  set(object_file "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.o")

  set(input_objects $<TARGET_OBJECTS:${objects_target_name}>)
  add_custom_command(
    OUTPUT ${object_file_raw}
    DEPENDS ${input_objects}
    COMMAND ${CMAKE_LINKER} -r ${input_objects} -o ${object_file_raw}
  )

  set(alias_attributes "0,function,global")
  if(ADD_ENTRYPOINT_OBJ_REDIRECTED)
    set(alias_attributes "${alias_attributes},hidden")
  endif()

  add_custom_command(
    OUTPUT ${object_file}
    # We llvm-objcopy here as GNU-binutils objcopy does not support the 'hidden' flag.
    DEPENDS ${object_file_raw} ${llvm-objcopy}
    COMMAND $<TARGET_FILE:llvm-objcopy> --add-symbol "${entrypoint_name}=.llvm.libc.entrypoint.${entrypoint_name}:${alias_attributes}" ${object_file_raw} ${object_file}
  )

  add_custom_target(
    ${fq_target_name}
    ALL
    DEPENDS ${object_file}
  )
  set(all_objects ${object_file})
  list(APPEND all_objects ${dep_objects})
  set(all_objects_raw ${object_file_raw})
  list(APPEND all_objects_raw ${dep_objects})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "TARGET_TYPE" ${ENTRYPOINT_OBJ_TARGET_TYPE}
      "OBJECT_FILES" "${all_objects}"
      "OBJECT_FILES_RAW" "${all_objects_raw}"
  )

  if(LLVM_LIBC_ENABLE_LINTING)
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
      COMMAND $<TARGET_FILE:clang-tidy> "--extra-arg=-fno-caret-diagnostics" --quiet
              # Path to directory containing compile_commands.json
              -p ${PROJECT_BINARY_DIR}
              ${ADD_ENTRYPOINT_OBJ_SRCS}
      # We have two options for running commands, add_custom_command and
      # add_custom_target. We don't want to run the linter unless source files
      # have changed. add_custom_target explicitly runs everytime therefore we
      # use add_custom_command. This function requires an output file and since
      # linting doesn't produce a file, we create a dummy file using a
      # crossplatform touch.
      COMMAND "${CMAKE_COMMAND}" -E touch ${lint_timestamp}
      COMMENT "Linting... ${target_name}"
      DEPENDS ${clang-tidy} ${objects_target_name} ${ADD_ENTRYPOINT_OBJ_SRCS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_custom_target(${fq_target_name}.__lint__
      DEPENDS ${lint_timestamp})
    add_dependencies(lint-libc ${fq_target_name}.__lint__)
    add_dependencies(${fq_target_name} ${fq_target_name}.__lint__)
  endif()

endfunction(add_entrypoint_object)

# A rule to build a library from a collection of entrypoint objects.
# Usage:
#     add_entrypoint_library(
#       DEPENDS <list of add_entrypoint_object targets>
#     )
function(add_entrypoint_library target_name)
  cmake_parse_arguments(
    "ENTRYPOINT_LIBRARY"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT ENTRYPOINT_LIBRARY_DEPENDS)
    message(FATAL_ERROR "'add_entrypoint_library' target requires a DEPENDS list of 'add_entrypoint_object' targets.")
  endif()

  set(obj_list "")
  foreach(dep IN LISTS ENTRYPOINT_LIBRARY_DEPENDS)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT (${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE}))
      message(FATAL_ERROR "Dependency '${dep}' of 'add_entrypoint_collection' is not an 'add_entrypoint_object' target.")
    endif()
    get_target_property(target_obj_files ${dep} "OBJECT_FILES")
    list(APPEND obj_list "${target_obj_files}")
  endforeach(dep)
  list(REMOVE_DUPLICATES obj_list)

  set(library_file "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${target_name}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  add_custom_command(
    OUTPUT ${library_file}
    COMMAND ${CMAKE_AR} -r ${library_file} ${obj_list}
    DEPENDS ${obj_list}
  )
  add_custom_target(
    ${target_name}
    ALL
    DEPENDS ${library_file}
  )
endfunction(add_entrypoint_library)

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
    OBJECT
    ${REDIRECTOR_OBJECT_SRC}
  )
  target_compile_options(
    ${target_name}
    BEFORE PRIVATE -fPIC
  )
endfunction(add_redirector_object)

# Rule to build a shared library of redirector objects.
function(add_redirector_library target_name)
  cmake_parse_arguments(
    "REDIRECTOR_LIBRARY"
    ""
    ""
    "DEPENDS"
    ${ARGN}
  )

  set(obj_files "")
  foreach(dep IN LISTS REDIRECTOR_LIBRARY_DEPENDS)
    # TODO: Ensure that each dep is actually a add_redirector_object target.
    list(APPEND obj_files $<TARGET_OBJECTS:${dep}>)
  endforeach(dep)

  # TODO: Call the linker explicitly instead of calling the compiler driver to
  # prevent DT_NEEDED on C++ runtime.
  add_library(
    ${target_name}
    SHARED
    ${obj_files}
  )
  set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_link_libraries(
    ${target_name}
    -nostdlib -lc -lm
  )

  set_target_properties(
    ${target_name}
    PROPERTIES
      LINKER_LANGUAGE "C"
  )
endfunction(add_redirector_library)

# Rule to add a libc unittest.
# Usage
#    add_libc_unittest(
#      <target name>
#      SUITE <name of the suite this test belongs to>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#      COMPILE_OPTIONS <list of special compile options for this target>
#    )
function(add_libc_unittest target_name)
  if(NOT LLVM_INCLUDE_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    "LIBC_UNITTEST"
    "" # No optional arguments
    "SUITE" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_UNITTEST_SRCS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a SRCS list of .cpp files.")
  endif()
  if(NOT LIBC_UNITTEST_DEPENDS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a DEPENDS list of 'add_entrypoint_object' targets.")
  endif()

  set(library_deps "")
  get_fq_deps_list(fq_deps_list ${LIBC_UNITTEST_DEPENDS})
  foreach(dep IN LISTS fq_deps_list)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      get_target_property(obj_files ${dep} "OBJECT_FILES_RAW")
      list(APPEND library_deps ${obj_files})
    elseif(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      get_target_property(obj_files ${dep} "OBJECT_FILES")
      list(APPEND library_deps ${obj_files})
    endif()
    # TODO: Check if the dep is a normal CMake library target. If yes, then add it
    # to the list of library_deps.
  endforeach(dep)
  list(REMOVE_DUPLICATES library_deps)

  get_fq_target_name(${target_name} fq_target_name)
  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_UNITTEST_SRCS}
    ${LIBC_UNITTEST_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )
  if(LIBC_UNITTEST_COMPILE_OPTIONS)
    target_compile_options(
      ${target_name}
      PRIVATE ${LIBC_UNITTEST_COMPILE_OPTIONS}
    )
  endif()

  if(library_deps)
    target_link_libraries(${fq_target_name} PRIVATE ${library_deps})
  endif()

  set_target_properties(${fq_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )

  target_link_libraries(${fq_target_name} PRIVATE LibcUnitTest libc_test_utils)

  add_custom_command(
    TARGET ${fq_target_name}
    POST_BUILD
    COMMAND $<TARGET_FILE:${fq_target_name}>
  )
  if(LIBC_UNITTEST_SUITE)
    add_dependencies(
      ${LIBC_UNITTEST_SUITE}
      ${fq_target_name}
    )
  endif()
endfunction(add_libc_unittest)

function(add_libc_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(check-libc ${suite_name})
endfunction(add_libc_testsuite)

# Rule to add a fuzzer test.
# Usage
#    add_libc_fuzzer(
#      <target name>
#      SRCS  <list of .cpp files for the test>
#      HDRS  <list of .h files for the test>
#      DEPENDS <list of dependencies>
#    )
function(add_libc_fuzzer target_name)
  cmake_parse_arguments(
    "LIBC_FUZZER"
    "" # No optional arguments
    "" # Single value arguments
    "SRCS;HDRS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )
  if(NOT LIBC_FUZZER_SRCS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a SRCS list of .cpp files.")
  endif()
  if(NOT LIBC_FUZZER_DEPENDS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a DEPENDS list of 'add_entrypoint_object' targets.")
  endif()

  get_fq_deps_list(fq_deps_list ${LIBC_FUZZER_DEPENDS})
  set(library_deps "")
  foreach(dep IN LISTS fq_deps_list)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if (dep_type)
      string(COMPARE EQUAL ${dep_type} ${ENTRYPOINT_OBJ_TARGET_TYPE} dep_is_entrypoint)
      if(dep_is_entrypoint)
        get_target_property(obj_file ${dep} "OBJECT_FILES_RAW")
        list(APPEND library_deps ${obj_file})
        continue()
      endif()
    endif()
    # TODO: Check if the dep is a normal CMake library target. If yes, then add it
    # to the list of library_deps.
  endforeach(dep)

  get_fq_target_name(${target_name} fq_target_name)
  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${LIBC_FUZZER_SRCS}
    ${LIBC_FUZZER_HDRS}
  )
  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )

  if(library_deps)
    target_link_libraries(${fq_target_name} PRIVATE ${library_deps})
  endif()

  set_target_properties(${fq_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )
  add_dependencies(libc-fuzzer ${fq_target_name})
endfunction(add_libc_fuzzer)

# Rule to add header only libraries.
# Usage
#    add_header_library(
#      <target name>
#      HDRS  <list of .h files part of the library>
#      DEPENDS <list of dependencies>
#    )
function(add_header_library target_name)
  cmake_parse_arguments(
    "ADD_HEADER"
    "" # No optional arguments
    "" # No Single value arguments
    "HDRS;DEPENDS" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_HEADER_HDRS)
    message(FATAL_ERROR "'add_header_library' target requires a HDRS list of .h files.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)

  set(FULL_HDR_PATHS "")
  # TODO: Remove this foreach block when we can switch to the new
  # version of the CMake policy CMP0076.
  foreach(hdr IN LISTS ADD_HEADER_HDRS)
    list(APPEND FULL_HDR_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/${hdr})
  endforeach()

  set(interface_target_name "${fq_target_name}_header_library__")

  add_library(${interface_target_name} INTERFACE)
  target_sources(${interface_target_name} INTERFACE ${FULL_HDR_PATHS})
  if(ADD_HEADER_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_HEADER_DEPENDS})
    add_dependencies(${interface_target_name} ${fq_deps_list})
  endif()

  add_custom_target(${fq_target_name})
  add_dependencies(${fq_target_name} ${interface_target_name})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "TARGET_TYPE" "HDR_LIBRARY"
  )
endfunction(add_header_library)
