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

# Enable the offload test suite distribution. This produces a portable
# install prefix containing all binaries and test files needed to run the
# HLSL offload test suite on another machine.
#
# Install with:
#   cmake --build build --target install-distribution
#   cmake --build build --target install-offload-tools
#   cmake --build build --target install-offload-test-suite
#
# Prerequisites on the target machine:
#   - Python 3.6+
#   - pip install lit pyyaml
#   - GPU drivers (D3D12, Vulkan, or Metal depending on test suite)
#   - For non-clang test suites: a DXC executable (clang-dxc is included)
#
# After installing, configure and run tests:
#   cd <prefix>/share/hlsl-test-suite
#   ./configure-test-suite.py --suite clang-d3d12
#   lit run/test/clang-d3d12
if (HLSL_ENABLE_OFFLOAD_DISTRIBUTION)
  # Lit utilities (FileCheck, split-file, etc.) require LLVM_INSTALL_UTILS.
  set(LLVM_INSTALL_UTILS ON CACHE BOOL "")
  set(LLVM_DISTRIBUTION_COMPONENTS
      "clang;hlsl-resource-headers;FileCheck;split-file;obj2yaml;not"
      CACHE STRING "")
endif()
