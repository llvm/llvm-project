set(LLVM_USE_SANITIZER "MemoryWithOrigins" CACHE STRING "")
set(LIBCXXABI_USE_LLVM_UNWINDER OFF CACHE BOOL "") # MSAN is compiled against the system unwinder, which leads to false positives
