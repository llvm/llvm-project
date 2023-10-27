add_custom_target(libc-integration-tests)
add_dependencies(check-libc libc-integration-tests)

function(add_libc_integration_test_suite name)
  add_custom_target(${name})
  add_dependencies(libc-integration-tests ${name})
endfunction()

add_subdirectory(startup)
add_subdirectory(scudo)
add_subdirectory(src)
