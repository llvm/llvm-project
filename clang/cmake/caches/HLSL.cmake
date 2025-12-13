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
