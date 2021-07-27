if(NOT LLVM_LIBC_INCLUDE_SCUDO)
  return()
endif()

add_executable(
  libc-scudo-integration-test
  integration_test.cpp
)

target_link_options(
  libc-scudo-integration-test
  PRIVATE
  # TODO: Uncomment "-nolibc" once llvm-libc is complete enough.
  # "-nolibc"
  -pthreads
)

target_link_libraries(libc-scudo-integration-test
  PRIVATE
  llvmlibc
)

add_executable(
  libc-gwp-asan-uaf-should-crash
  gwp_asan_should_crash.cpp
)

target_link_options(
  libc-gwp-asan-uaf-should-crash
  PRIVATE
  # TODO: Uncomment "-nolibc" once llvm-libc is complete enough.
  # "-nolibc"
  -pthreads
)

target_link_libraries(libc-gwp-asan-uaf-should-crash
  PRIVATE
  llvmlibc
)
