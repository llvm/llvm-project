# A rule to add loader tests. When we have a complete loader, we should
# be able to use the add_libc_unittest rule or an extension of it. But,
# while the loader is getting built, we need to use a special rule like
# this.
function(add_loader_test target_name)
  if(NOT CMAKE_HOST_UNIX)
    message(
        WARNING
        "Loader tests currently assume a POSIX/Unix like environment and "
        "may not work on your platform.")
  endif()

  cmake_parse_arguments(
    "ADD_LOADER_TEST"
    ""    # No option arguments
    "SRC" # Single value arguments
    "DEPENDS;ARGS;ENV" # Multivalue arguments.
    ${ARGN}
  )

  get_fq_target_name(${target_name} fq_target_name)
  add_executable(
    ${fq_target_name}
    EXCLUDE_FROM_ALL
    ${ADD_LOADER_TEST_SRC}
  )

  set_target_properties(${fq_target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_include_directories(
    ${fq_target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )

  if(ADD_LOADER_TEST_DEPENDS)
    get_fq_deps_list(fq_deps_list ${ADD_LOADER_TEST_DEPENDS})
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
    COMMAND ${ADD_LOADER_TEST_ENV} $<TARGET_FILE:${fq_target_name}> ${ADD_LOADER_TEST_ARGS}
  )

  add_dependencies(libc_loader_tests ${fq_target_name})
endfunction(add_loader_test)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()
