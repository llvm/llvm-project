if(NOT LIBC_TARGET_OS_IS_GPU)
  message(FATAL_ERROR
          "libc build: Invalid attempt to set up GPU architectures.")
endif()

# Ensure the compiler is a valid clang when building the GPU target.
set(req_ver "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(LLVM_VERSION_MAJOR AND NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND
   ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "${req_ver}"))
  message(FATAL_ERROR "Cannot build libc for GPU. CMake compiler "
                      "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' "
                      " is not 'Clang ${req_ver}'.")
endif()
if(NOT LLVM_LIBC_FULL_BUILD)
  message(FATAL_ERROR "LLVM_LIBC_FULL_BUILD must be enabled to build libc for "
                      "GPU.")
endif()

# Identify the program used to package multiple images into a single binary.
get_filename_component(compiler_path ${CMAKE_CXX_COMPILER} DIRECTORY)
find_program(LIBC_CLANG_OFFLOAD_PACKAGER
             NAMES clang-offload-packager NO_DEFAULT_PATH
             PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
if(NOT LIBC_CLANG_OFFLOAD_PACKAGER)
  message(FATAL_ERROR "Cannot find the 'clang-offload-packager' for the GPU "
                      "build")
endif()

# Identify llvm-link program so we can merge the output IR into a single blob.
find_program(LIBC_LLVM_LINK
             NAMES llvm-link NO_DEFAULT_PATH
             PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
if(NOT LIBC_LLVM_LINK)
  message(FATAL_ERROR "Cannot find 'llvm-link' for the GPU build")
endif()

# Optionally set up a job pool to limit the number of GPU tests run in parallel.
# This is sometimes necessary as running too many tests in parallel can cause
# the GPU or driver to run out of resources.
set(LIBC_GPU_TEST_JOBS "" CACHE STRING "Number of jobs to use for GPU tests")
if(LIBC_GPU_TEST_JOBS)
  set_property(GLOBAL PROPERTY JOB_POOLS LIBC_GPU_TEST_POOL=${LIBC_GPU_TEST_JOBS})
  set(LIBC_HERMETIC_TEST_JOB_POOL JOB_POOL LIBC_GPU_TEST_POOL)
else()
  set_property(GLOBAL PROPERTY JOB_POOLS LIBC_GPU_TEST_POOL=1)
  set(LIBC_HERMETIC_TEST_JOB_POOL JOB_POOL LIBC_GPU_TEST_POOL)
endif()

set(LIBC_GPU_TEST_ARCHITECTURE "" CACHE STRING "Architecture for the GPU tests")
if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  check_cxx_compiler_flag("-nogpulib -mcpu=native" PLATFORM_HAS_GPU)
elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  # Identify any locally installed NVIDIA GPUs on the system using 'nvptx-arch'.
  # Using 'check_cxx_compiler_flag' does not work currently due to the link job.
  find_program(LIBC_NVPTX_ARCH
               NAMES nvptx-arch NO_DEFAULT_PATH
               PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  if(LIBC_NVPTX_ARCH)
    execute_process(COMMAND ${LIBC_NVPTX_ARCH}
                    OUTPUT_VARIABLE arch_tool_output
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(arch_tool_output MATCHES "^sm_[0-9]+")
      set(PLATFORM_HAS_GPU TRUE)
    endif()
  endif()
endif()

set(gpu_test_architecture "")
if(LIBC_GPU_TEST_ARCHITECTURE)
  set(LIBC_GPU_TESTS_DISABLED FALSE)
  set(gpu_test_architecture ${LIBC_GPU_TEST_ARCHITECTURE})
  message(STATUS "Using user-specified GPU architecture for testing: "
                 "'${gpu_test_architecture}'")
elseif(PLATFORM_HAS_GPU)
  set(LIBC_GPU_TESTS_DISABLED FALSE)
  set(gpu_test_architecture "native")
  message(STATUS "Using GPU architecture detected on the system for testing: "
                 "'native'")
else()
  set(LIBC_GPU_TESTS_DISABLED TRUE)
  message(STATUS "No GPU architecture detected or provided, tests will not be "
                 "built")
endif()
set(LIBC_GPU_TARGET_ARCHITECTURE "${gpu_test_architecture}")

if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  # FIXME: This is a hack required to keep the CUDA package from trying to find
  #        pthreads. We only link the CUDA driver, so this is unneeded.
  add_library(CUDA::cudart_static_deps IMPORTED INTERFACE)

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    get_filename_component(LIBC_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY ABSOLUTE)
  endif()
endif()

if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  # The AMDGPU environment uses different code objects to encode the ABI for
  # kernel calls and intrinsic functions. We want to specify this manually to
  # conform to whatever the test suite was built to handle.
  set(LIBC_GPU_CODE_OBJECT_VERSION 5)
endif()
