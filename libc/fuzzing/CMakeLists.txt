set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
add_custom_target(libc-fuzzer)

add_subdirectory(__support)
# TODO(#85680): Re-enable math fuzzing after headers are sorted out
# add_subdirectory(math)
add_subdirectory(stdlib)
add_subdirectory(stdio)
add_subdirectory(string)
