set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
add_custom_target(libc-fuzzer)
add_dependencies(check-libc libc-fuzzer)

add_subdirectory(math)
add_subdirectory(string)
