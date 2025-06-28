set(LLVM_USE_SANITIZER "Thread" CACHE STRING "")
set(LIBCXXABI_USE_LLVM_UNWINDER OFF CACHE BOOL "") # TSAN is compiled against the system unwinder, which leads to false positives
set(LIBCXX_INCLUDE_BENCHMARKS OFF CACHE BOOL "FIXME: This is a temporary workaround to get tsan running again")