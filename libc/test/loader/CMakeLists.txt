add_custom_target(libc_loader_tests)

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

  add_executable(
    ${target_name}
    EXCLUDE_FROM_ALL
    ${ADD_LOADER_TEST_SRC}
  )

  set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set(dep_objects "")
  if(ADD_LOADER_TEST_DEPENDS)
    add_dependencies(${target_name} ${ADD_LOADER_TEST_DEPENDS})
    foreach(dep IN LISTS ADD_LOADER_TEST_DEPENDS)
      get_target_property(objfile ${dep} "OBJECT_FILE")
      if(NOT objfile)
        message(
            FATAL_ERROR
            "Unexpected dependency of an `add_loader_test` target. A dependency "
            "should be a target of type `add_entrypoint_object, `add_object`, or "
            "`add_loader_object`.")
      endif()
      list(APPEND dep_objects ${objfile})
    endforeach(dep)
  endif()

  target_include_directories(
    ${target_name}
    PRIVATE
      ${LIBC_SOURCE_DIR}
      ${LIBC_BUILD_DIR}
      ${LIBC_BUILD_DIR}/include
  )

  target_link_libraries(${target_name} ${dep_objects})

  target_link_options(
    ${target_name}
    BEFORE PRIVATE
    -nostdlib
  )

  add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND ${ADD_LOADER_TEST_ENV} $<TARGET_FILE:${target_name}> ${ADD_LOADER_TEST_ARGS}
  )

  add_dependencies(libc_loader_tests ${target_name})
endfunction(add_loader_test)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${LIBC_TARGET_OS})
  add_subdirectory(${LIBC_TARGET_OS})
endif()
