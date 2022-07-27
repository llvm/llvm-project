add_libc_unittest(
  utils_test
  SUITE
    libc_string_unittests
  SRCS
    address_test.cpp
    backend_test.cpp
    elements_test.cpp
    memory_access_test.cpp
    utils_test.cpp
  COMPILE_OPTIONS
    ${LIBC_COMPILE_OPTIONS_NATIVE}
    -ffreestanding
    -pthread
  DEPENDS
    libc.src.string.memory_utils.memory_utils
    libc.src.__support.CPP.array
    libc.src.__support.CPP.array_ref
)

if(NOT LLVM_LIBC_FULL_BUILD)
# Disabling this unittest in fullbuild mode as #include<sstream> is pulling an
# incomplete pthread implementation from llvm-libc.
add_libc_unittest(
  algorithm_test
  SUITE
    libc_string_unittests
  SRCS
    algorithm_test.cpp
  DEPENDS
    libc.src.string.memory_utils.memory_utils
    libc.src.__support.CPP.array
    libc.src.__support.CPP.array_ref
)
endif()
