# A rule to add startup system tests. When we have a complete startup system,
# we should be able to use the add_libc_unittest rule or an extension of it.
# But, while the system is being developed,  we need to use a special rule like
# this.
function(add_startup_test target_name)
  if(NOT CMAKE_HOST_UNIX)
    message(
        WARNING
        "Test for the startup system currently assume a POSIX/Unix like "
        "environment and may not work on your platform.")
  endif()

  cmake_parse_arguments(
    "ADD_STARTUP_TEST"
    ""    # No option arguments
    "SRC" # Single value arguments
    "DEPENDS;ARGS;ENV" # Multivalue arguments.
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)
  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${ADD_STARTUP_TEST_SRC}
  )

  set_target_properties(${fq_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_INCLUDE_DIR}
  )

  if(ADD_STARTUP_TEST_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_STARTUP_TEST_DEPENDS})
    add_dependencies(${fq_target_name} ${fq_deps_list})
    get_object_files_for_test(link_object_files has_skipped_entrypoint_list ${fq_deps_list})
    target_link_libraries(${fq_target_name} ${link_object_files})
  endif()

  target_link_options(
    ${fq_target_name}
    BEFORE PRIVATE
    -nostdlib
  )

  add_custom_command(
    TARGET ${fq_target_name}
    POST_BUILD
    COMMAND ${ADD_STARTUP_TEST_ENV} $<TARGET_FILE:${fq_target_name}> ${ADD_STARTUP_TEST_ARGS}
  )

  add_dependencies(libc_startup_tests ${fq_target_name})
endfunction(add_startup_test)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()
