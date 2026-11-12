set(LLVM_USE_SANITIZER "Thread" CACHE STRING "")
set(LIBCXXABI_USE_LLVM_UNWINDER OFF CACHE BOOL "") # TSAN is compiled against the system unwinder, which leads to false positives
set(LIBCXX_TEST_PARAMS "enable_modules=clang" CACHE STRING "")
set(LIBCXXABI_TEST_PARAMS "${LIBCXX_TEST_PARAMS}" CACHE STRING "")
