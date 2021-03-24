# This is a helper function and not a build rule. It is to be used by the
# various test rules to generate the full list of object files
# recursively produced by "add_entrypoint_object" and "add_object_library"
# targets.
# Usage:
#   get_object_files_for_test(<result var>
#                             <skipped_entrypoints_var>
#                             <target0> [<target1> ...])
#
#   The list of object files is collected in <result_var>.
#   If skipped entrypoints were found, then <skipped_entrypoints_var> is
#   set to a true value.
#   targetN is either an "add_entrypoint_target" target or an
#   "add_object_library" target.
function(get_object_files_for_test result skipped_entrypoints_list)
  set(object_files "")
  set(skipped_list "")
  foreach(dep IN LISTS ARGN)
    get_target_property(dep_type ${dep} "TARGET_TYPE")
    if(NOT dep_type)
      # Target for which TARGET_TYPE property is not set do not
      # provide any object files.
      continue()
    endif()

    if(${dep_type} STREQUAL ${OBJECT_LIBRARY_TARGET_TYPE})
      get_target_property(dep_object_files ${dep} "OBJECT_FILES")
      if(dep_object_files)
        list(APPEND object_files ${dep_object_files})
      endif()
    elseif(${dep_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE})
      get_target_property(is_skipped ${dep} "SKIPPED")
      if(is_skipped)
        list(APPEND skipped_list ${dep})
        continue()
      endif()
      get_target_property(object_file_raw ${dep} "OBJECT_FILE_RAW")
      if(object_file_raw)
        list(APPEND object_files ${object_file_raw})
      endif()
    endif()

    get_target_property(indirect_deps ${dep} "DEPS")
    get_object_files_for_test(
        indirect_objfiles indirect_skipped_list ${indirect_deps})
    list(APPEND object_files ${indirect_objfiles})
    if(indirect_skipped_list)
      list(APPEND skipped_list ${indirect_skipped_list})
    endif()
  endforeach(dep)
  list(REMOVE_DUPLICATES object_files)
  set(${result} ${object_files} PARENT_SCOPE)
  list(REMOVE_DUPLICATES skipped_list)
  set(${skipped_entrypoints_list} ${skipped_list} PARENT_SCOPE)
endfunction(get_object_files_for_test)

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
    message(FATAL_ERROR "'add_libc_unittest' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_UNITTEST_DEPENDS)
    message(FATAL_ERROR "'add_libc_unittest' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${LIBC_UNITTEST_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    # If a test is OS/target machine independent, it has to be skipped if the
    # OS/target machine combination does not provide any dependent entrypoints.
    # If a test is OS/target machine specific, then such a test will live is a
    # OS/target machine specific directory and will be skipped at the directory
    # level if required.
    #
    # There can potentially be a setup like this: A unittest is setup for a
    # OS/target machine independent object library, which in turn depends on a
    # machine specific object library. Such a test would be testing internals of
    # the libc and it is assumed that they will be rare in practice. So, they
    # can be skipped in the corresponding CMake files using platform specific
    # logic. This pattern is followed in the loader tests for example.
    #
    # Another pattern that is present currently is to detect machine
    # capabilities and add entrypoints and tests accordingly. That approach is
    # much lower level approach and is independent of the kind of skipping that
    # is happening here at the entrypoint level.

    set(msg "Skipping unittest ${fq_target_name} as it has missing deps: "
            "${skipped_entrypoints_list}.")
    message(STATUS ${msg})
    return()
  endif()

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
      ${fq_target_name}
      PRIVATE ${LIBC_UNITTEST_COMPILE_OPTIONS}
    )
  endif()

  target_link_libraries(${fq_target_name} PRIVATE ${link_object_files})

  set_target_properties(${fq_target_name}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

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
  add_dependencies(check-llvmlibc ${suite_name})
endfunction(add_libc_testsuite)

function(add_libc_exhaustive_testsuite suite_name)
  add_custom_target(${suite_name})
  add_dependencies(exhaustive-check-libc ${suite_name})
endfunction(add_libc_exhaustive_testsuite)

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
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a SRCS list of .cpp "
                        "files.")
  endif()
  if(NOT LIBC_FUZZER_DEPENDS)
    message(FATAL_ERROR "'add_libc_fuzzer' target requires a DEPENDS list of "
                        "'add_entrypoint_object' targets.")
  endif()

  get_fq_target_name(${target_name} fq_target_name)
  get_fq_deps_list(fq_deps_list ${LIBC_FUZZER_DEPENDS})
  get_object_files_for_test(
      link_object_files skipped_entrypoints_list ${fq_deps_list})
  if(skipped_entrypoints_list)
    set(msg "Skipping fuzzer target ${fq_target_name} as it has missing deps: "
            "${skipped_entrypoints_list}.")
    message(STATUS ${msg})
    add_custom_target(${fq_target_name})

    # A post build custom command is used to avoid running the command always.
    add_custom_command(
      TARGET ${fq_target_name}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E echo ${msg}
    )
    return()
  endif()

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

  target_link_libraries(${fq_target_name} PRIVATE ${link_object_files})

  set_target_properties(${fq_target_name}
      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_dependencies(
    ${fq_target_name}
    ${fq_deps_list}
  )
  add_dependencies(libc-fuzzer ${fq_target_name})
endfunction(add_libc_fuzzer)
