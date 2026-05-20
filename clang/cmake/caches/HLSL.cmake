# Including the native target is important because some of LLVM's tests fail if
# you don't.
set(LLVM_TARGETS_TO_BUILD "Native;SPIRV" CACHE STRING "")

# Include the DirectX target for DXIL code generation.
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "DirectX" CACHE STRING "")

set(LLVM_ENABLE_PROJECTS "clang;clang-tools-extra" CACHE STRING "")

set(CLANG_ENABLE_HLSL On CACHE BOOL "")

if (HLSL_ENABLE_DISTRIBUTION)
  set(LLVM_DISTRIBUTION_COMPONENTS
      "clang;hlsl-resource-headers;clangd"
      CACHE STRING "")
endif()

# Enable the offload test suite distribution. Produces a portable install
# prefix containing the binaries, headers, and libs needed to run the HLSL
# offload test suite on another machine. See the offload-test-suite repo
# (docs/offload-distribution.md) for setup, prerequisites, and run
# instructions.
if (HLSL_ENABLE_OFFLOAD_DISTRIBUTION)
  if (NOT "OffloadTest" IN_LIST LLVM_EXTERNAL_PROJECTS)
    message(FATAL_ERROR
      "HLSL_ENABLE_OFFLOAD_DISTRIBUTION requires OffloadTest to be enabled "
      "as an external project. Pass -DLLVM_EXTERNAL_PROJECTS=OffloadTest "
      "and -DLLVM_EXTERNAL_OFFLOADTEST_SOURCE_DIR=<path-to-offload-test-suite>.")
  endif()
  # Lit utilities (FileCheck, split-file, etc.) require LLVM_INSTALL_UTILS.
  set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
  set(LLVM_DISTRIBUTION_COMPONENTS
      "clang;hlsl-resource-headers;FileCheck;split-file;obj2yaml;not"
      CACHE STRING "")
endif()
