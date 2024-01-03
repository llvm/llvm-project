# Plain options configure the first build.
# BOOTSTRAP_* options configure the second build.
# BOOTSTRAP_BOOTSTRAP_* options configure the third build.

set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "")

# Stage 1 Bootstrap Setup
set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
set(CLANG_BOOTSTRAP_TARGETS
  clang
  check-all
  check-llvm
  check-clang
  test-suite
  stage3
  stage3-clang
  stage3-check-all
  stage3-check-llvm
  stage3-check-clang
  stage3-install
  stage3-test-suite CACHE STRING "")

# Stage 1 Options
set(LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

# Stage 2 Bootstrap Setup
set(BOOTSTRAP_CLANG_ENABLE_BOOTSTRAP ON CACHE STRING "")
set(BOOTSTRAP_CLANG_BOOTSTRAP_TARGETS
  clang
  check-all
  check-llvm
  check-clang CACHE STRING "")

# Stage 2 Options
set(BOOTSTRAP_LLVM_ENABLE_PROJECTS "clang" CACHE STRING "")
set(BOOTSTRAP_LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

# Stage 3 Options
set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "")
set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_PROJECTS "clang;lld;lldb;clang-tools-extra;bolt;polly;mlir;flang" CACHE STRING "")
