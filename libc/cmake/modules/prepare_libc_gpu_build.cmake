if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  message(FATAL_ERROR
          "libc build: Invalid attempt to set up GPU architectures.")
endif()

# Set up the target architectures to build the GPU libc for.
set(all_amdgpu_architectures "gfx700;gfx701;gfx801;gfx803;gfx900;gfx902;gfx906"
                             "gfx908;gfx90a;gfx90c;gfx940;gfx1010;gfx1030"
                             "gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
                             "gfx1100;gfx1101;gfx1102;gfx1103")
set(all_nvptx_architectures "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62"
                            "sm_70;sm_72;sm_75;sm_80;sm_86")
set(all_gpu_architectures
    "${all_amdgpu_architectures};${all_nvptx_architectures}")
set(LIBC_GPU_ARCHITECTURES "all" CACHE STRING
    "List of GPU architectures to build the libc for.")

# Ensure the compiler is a valid clang when building the GPU target.
set(req_ver "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "${req_ver}"))
  message(FATAL_ERROR "Cannot build libc for GPU. CMake compiler "
                      "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' "
                      " is not 'Clang ${req_ver}'.")
endif()
if(NOT LLVM_LIBC_FULL_BUILD)
  message(FATAL_ERROR "LLVM_LIBC_FULL_BUILD must be enabled to build libc for "
                      "GPU.")
endif()

# Identify any locally installed AMD GPUs on the system using 'amdgpu-arch'.
find_program(LIBC_AMDGPU_ARCH
             NAMES amdgpu-arch
             PATHS ${LLVM_BINARY_DIR}/bin /opt/rocm/llvm/bin/)

# Identify any locally installed NVIDIA GPUs on the system using 'nvptx-arch'.
find_program(LIBC_NVPTX_ARCH
             NAMES nvptx-arch
             PATHS ${LLVM_BINARY_DIR}/bin)

# Get the list of all natively supported GPU architectures.
set(detected_gpu_architectures "")
foreach(arch_tool ${LIBC_NVPTX_ARCH} ${LIBC_AMDGPU_ARCH})
  if(arch_tool)
    execute_process(COMMAND ${arch_tool}
                    OUTPUT_VARIABLE arch_tool_output
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "\n" ";" arch_list "${arch_tool_output}")
    list(APPEND detected_gpu_architectures "${arch_list}")
  endif()
endforeach()
list(REMOVE_DUPLICATES detected_gpu_architectures)

if(LIBC_GPU_ARCHITECTURES STREQUAL "all")
  set(LIBC_GPU_ARCHITECTURES ${all_gpu_architectures})
elseif(LIBC_GPU_ARCHITECTURES STREQUAL "native")
  if(NOT detected_gpu_architectures)
    message(FATAL_ERROR "No GPUs found on the system when using 'native'")
  endif()
  set(LIBC_GPU_ARCHITECTURES ${detected_gpu_architectures})
endif()
message(STATUS "Building libc for the following GPU architecture(s): "
               "${LIBC_GPU_ARCHITECTURES}")

# Identify the program used to package multiple images into a single binary.
find_program(LIBC_CLANG_OFFLOAD_PACKAGER
             NAMES clang-offload-packager
             PATHS ${LLVM_BINARY_DIR}/bin)
if(NOT LIBC_CLANG_OFFLOAD_PACKAGER)
  message(FATAL_ERROR "Cannot find the 'clang-offload-packager' for the GPU "
                      "build")
endif()

set(LIBC_GPU_TEST_ARCHITECTURE "" CACHE STRING "Architecture for the GPU tests")

set(gpu_test_architecture "")
if(LIBC_GPU_TEST_ARCHITECTURE)
  set(gpu_test_architecture ${LIBC_GPU_TEST_ARCHITECTURE})
  message(STATUS "Using user-specified GPU architecture for testing: "
                 "'${gpu_test_architecture}'")
elseif(detected_gpu_architectures)
  list(GET detected_gpu_architectures 0 gpu_test_architecture)
  message(STATUS "Using GPU architecture detected on the system for testing: "
                 "'${gpu_test_architecture}'")
else()
  message(STATUS "No GPU architecture set for testing. GPU tests will not be "
                 "availibe. Set 'LIBC_GPU_TEST_ARCHITECTURE' to override.")
  return()
endif()

if("${gpu_test_architecture}" IN_LIST all_amdgpu_architectures)
  set(LIBC_GPU_TARGET_ARCHITECTURE_IS_AMDGPU TRUE)
  set(LIBC_GPU_TARGET_TRIPLE "amdgcn-amd-amdhsa")
  set(LIBC_GPU_TARGET_ARCHITECTURE "${gpu_test_architecture}")
elseif("${gpu_test_architecture}" IN_LIST all_nvptx_architectures)
  set(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX TRUE)
  set(LIBC_GPU_TARGET_TRIPLE "nvptx64-nvidia-cuda")
  set(LIBC_GPU_TARGET_ARCHITECTURE "${gpu_test_architecture}")
else()
  message(FATAL_ERROR "Unknown GPU architecture '${gpu_test_architecture}'")
endif()

if(LIBC_GPU_TARGET_ARCHITECTURE_IS_NVPTX)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    get_filename_component(LIBC_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY ABSOLUTE)
  endif()
endif()
