add_custom_target(libc_linux_tests)
add_dependencies(check_libc libc_linux_tests)

add_subdirectory(x86_64)
