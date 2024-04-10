# Plain options configure the first build.
# BOOTSTRAP_* options configure the second build.
# BOOTSTRAP_BOOTSTRAP_* options configure the third build.

# General Options
set(LLVM_RELEASE_ENABLE_LTO THIN CACHE STRING "")
set(LLVM_RELEASE_ENABLE_PGO ON CACHE BOOL "")

set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "")

# Stage 1 Bootstrap Setup
set(CLANG_ENABLE_BOOTSTRAP ON CACHE BOOL "")
if (LLVM_RELEASE_ENABLE_PGO)
  set(CLANG_BOOTSTRAP_TARGETS
    generate-profdata
    stage2
    stage2-clang
    stage2-distribution
    stage2-install
    stage2-install-distribution
    stage2-install-distribution-toolchain
    stage2-check-all
    stage2-check-llvm
    stage2-check-clang
    stage2-test-suite CACHE STRING "")
else()
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
endif()

# Stage 1 Options
set(STAGE1_PROJECTS "clang")
set(STAGE1_RUNTIMES "")

if (LLVM_RELEASE_ENABLE_PGO)
  list(APPEND STAGE1_PROJECTS "lld")
  list(APPEND STAGE1_RUNTIMES "compiler-rt")
endif()

set(LLVM_ENABLE_RUNTIMES ${STAGE1_RUNTIMES} CACHE STRING "")
set(LLVM_ENABLE_PROJECTS ${STAGE1_PROJECTS} CACHE STRING "")

set(LLVM_TARGETS_TO_BUILD Native CACHE STRING "")

# Stage 2 Bootstrap Setup
set(BOOTSTRAP_CLANG_ENABLE_BOOTSTRAP ON CACHE STRING "")
set(BOOTSTRAP_CLANG_BOOTSTRAP_TARGETS
  clang
  check-all
  check-llvm
  check-clang CACHE STRING "")

# Stage 2 Options
set(STAGE2_PROJECTS "clang")
set(STAGE2_RUNTIMES "")

if (LLVM_RELEASE_ENABLE_LTO OR LLVM_RELEASE_ENABLE_PGO)
 list(APPEND STAGE2_PROJECTS "lld")
endif()

if (LLVM_RELEASE_ENABLE_PGO)
  set(BOOTSTRAP_LLVM_BUILD_INSTRUMENTED IR CACHE STRING "")
  list(APPEND STAGE2_RUNTIMES "compiler-rt")
  set(BOOTSTRAP_LLVM_ENABLE_LTO ${LLVM_RELEASE_ENABLE_LTO})
  if (LLVM_RELEASE_ENABLE_LTO)
    set(BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
  endif()
endif()

set(BOOTSTRAP_LLVM_ENABLE_PROJECTS ${STAGE2_PROJECTS} CACHE STRING "")
set(BOOTSTRAP_LLVM_ENABLE_RUNTIMES ${STAGE2_RUNTIMES} CACHE STRING "")
if (NOT LLVM_RELEASE_ENABLE_PGO)
  set(BOOTSTRAP_LLVM_TARGETS_TO_BUILD Native CACHE STRING "")
endif()

# Stage 3 Options
set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_RUNTIMES "compiler-rt;libcxx;libcxxabi;libunwind" CACHE STRING "")
set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_PROJECTS "clang;lld;lldb;clang-tools-extra;bolt;polly;mlir;flang" CACHE STRING "")
set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_LTO ${LLVM_RELEASE_ENABLE_LTO} CACHE STRING "")
if (LLVM_RELEASE_ENABLE_LTO)
  set(BOOTSTRAP_BOOTSTRAP_LLVM_ENABLE_LLD ON CACHE BOOL "")
endif()
