# This file sets up a CMakeCache for the ROCm LLVM toolchain shipped by TheRock.
#
#   compiler  amdllvm, which provides amdclang, amdflang, and the other amd*
#             tools, and the ROCm device libraries.
#   dev       The SPIR-V translator's library, headers, and pkg-config file.
#   tools     amd-llvm-spirv, the SPIR-V translator, and sqtt-marker, an LLVM
#             plugin that inserts thread trace markers. LLVM plugins do not
#             work on Windows.
#
# It also builds the host compiler-rt for i386 and enables GPU support in the
# host sanitizers.
#
# Options read by this file must be passed before -C:
#   LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR
#       Required. Source of the SPIR-V translator, ROCm/SPIRV-LLVM-Translator.
#   SANITIZER_HSA_INCLUDE_PATH
#       Directory containing hsa.h. Enables GPU support in the host sanitizers.

if(NOT LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR)
  message(FATAL_ERROR "LLVM_EXTERNAL_SPIRV_LLVM_TRANSLATOR_SOURCE_DIR must point to the SPIR-V translator source")
endif()

get_filename_component(_THEROCK_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

include(${CMAKE_CURRENT_LIST_DIR}/ROCm.cmake)

# amdclang, amdclang++, amdflang, amdlld, ...
set(CLANG_ENABLE_AMDCLANG ON CACHE BOOL "")

set(LLVM_EXTERNAL_ROCM_DEVICE_LIBS_SOURCE_DIR "${_THEROCK_ROOT}/amd/device-libs" CACHE PATH "")
set(LLVM_EXTERNAL_SQTT_MARKER_SOURCE_DIR "${_THEROCK_ROOT}/amd/sqtt-marker" CACHE PATH "")
if(CMAKE_HOST_WIN32)
  set(LLVM_EXTERNAL_PROJECTS "rocm-device-libs;spirv-llvm-translator" CACHE STRING "")
else()
  set(LLVM_EXTERNAL_PROJECTS "rocm-device-libs;spirv-llvm-translator;sqtt-marker" CACHE STRING "")
endif()

set_property(CACHE LLVM_compiler_DISTRIBUTION_COMPONENTS APPEND PROPERTY VALUE
  amdllvm
  device-libs)
# The translator's library, LLVMSPIRVAMDLib, is in llvm-libraries.
set_property(CACHE LLVM_dev_DISTRIBUTION_COMPONENTS APPEND PROPERTY VALUE
  spirv-llvm-translator)
set_property(CACHE LLVM_tools_DISTRIBUTION_COMPONENTS APPEND PROPERTY VALUE
  amd-llvm-spirv)

if(NOT CMAKE_HOST_WIN32)
  set_property(CACHE LLVM_tools_DISTRIBUTION_COMPONENTS APPEND PROPERTY VALUE
    sqtt-marker)

  # Builds compiler-rt for every architecture the host compiler supports.
  set(BUILTINS_${ROCM_HOST_TRIPLE}_COMPILER_RT_DEFAULT_TARGET_ONLY OFF CACHE BOOL "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_COMPILER_RT_DEFAULT_TARGET_ONLY OFF CACHE BOOL "")

  # libomp is loaded from several depths below <rocm>, e.g. lib/llvm/lib/<triple>.
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBOMP_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib:$ORIGIN/../../../lib:$ORIGIN/../../../../lib" CACHE STRING "")
  set(RUNTIMES_${ROCM_HOST_TRIPLE}_LIBOMPTARGET_NO_SANITIZER_AMDGPU ON CACHE BOOL "")
  if(SANITIZER_HSA_INCLUDE_PATH)
    set(RUNTIMES_${ROCM_HOST_TRIPLE}_SANITIZER_AMDGPU ON CACHE BOOL "")
    set(RUNTIMES_${ROCM_HOST_TRIPLE}_SANITIZER_HSA_INCLUDE_PATH "${SANITIZER_HSA_INCLUDE_PATH}" CACHE PATH "")
    set(RUNTIMES_${ROCM_HOST_TRIPLE}_SANITIZER_COMGR_INCLUDE_PATH "${_THEROCK_ROOT}/amd/comgr/include" CACHE PATH "")
  endif()
endif()
