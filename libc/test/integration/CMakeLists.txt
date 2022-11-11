add_custom_target(libc-integration-tests)

function(add_libc_integration_test_suite name)
  add_custom_target(${name})
  add_dependencies(libc-integration-tests ${name})
endfunction()

add_library(
  libc_integration_test_dummy
  STATIC
  dummy.cpp
)
set_target_properties(libc_integration_test_dummy
    PROPERTIES
      ARCHIVE_OUTPUT_NAME dummy
      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(loader)
add_subdirectory(scudo)
add_subdirectory(src)
