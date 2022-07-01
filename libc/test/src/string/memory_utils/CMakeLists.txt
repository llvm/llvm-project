# These tests are only exercising implementation details
#  - They don't need to be compiled with '-ffreestanding'
#  - They can use C++ facilities like <iostream>.
add_libc_unittest(
  implementation_details_test
  SUITE
    libc_string_unittests
  SRCS
    address_test.cpp
    algorithm_test.cpp
    backend_test.cpp
    elements_test.cpp
    memory_access_test.cpp
    utils_test.cpp
  COMPILE_OPTIONS
    ${LIBC_COMPILE_OPTIONS_NATIVE}
  DEPENDS
    libc.src.string.memory_utils.memory_utils
    libc.src.__support.CPP.array
    libc.src.__support.CPP.array_ref
)
