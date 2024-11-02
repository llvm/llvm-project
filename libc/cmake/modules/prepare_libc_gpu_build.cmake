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
if(TARGET clang-offload-packager)
  get_target_property(LIBC_CLANG_OFFLOAD_PACKAGER clang-offload-packager LOCATION)
else()
  find_program(LIBC_CLANG_OFFLOAD_PACKAGER
               NAMES clang-offload-packager NO_DEFAULT_PATH
               PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  if(NOT LIBC_CLANG_OFFLOAD_PACKAGER)
    message(FATAL_ERROR "Cannot find the 'clang-offload-packager' for the GPU "
                        "build")
  endif()
endif()

# Identify llvm-link program so we can merge the output IR into a single blob.
if(TARGET llvm-link)
  get_target_property(LIBC_LLVM_LINK llvm-link LOCATION)
else()
  find_program(LIBC_LLVM_LINK
               NAMES llvm-link NO_DEFAULT_PATH
               PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  if(NOT LIBC_LLVM_LINK)
    message(FATAL_ERROR "Cannot find 'llvm-link' for the GPU build")
  endif()
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
  # Identify any locally installed AMD GPUs on the system using 'amdgpu-arch'.
  if(TARGET amdgpu-arch)
    get_target_property(LIBC_AMDGPU_ARCH amdgpu-arch LOCATION)
  else()
    find_program(LIBC_AMDGPU_ARCH
                 NAMES amdgpu-arch NO_DEFAULT_PATH
                 PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  endif()
  if(LIBC_AMDGPU_ARCH)
    execute_process(COMMAND ${LIBC_AMDGPU_ARCH}
                    OUTPUT_VARIABLE arch_tool_output
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(arch_tool_output MATCHES "^gfx[0-9]+")
      set(PLATFORM_HAS_GPU TRUE)
    endif()
  endif()
elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  # Identify any locally installed NVIDIA GPUs on the system using 'nvptx-arch'.
  if(TARGET nvptx-arch)
    get_target_property(LIBC_NVPTX_ARCH nvptx-arch LOCATION)
  else()
    find_program(LIBC_NVPTX_ARCH
                 NAMES nvptx-arch NO_DEFAULT_PATH
                 PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  endif()
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
if(DEFINED LLVM_TARGETS_TO_BUILD AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU
   AND NOT "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(LIBC_GPU_TESTS_DISABLED TRUE)
  message(STATUS "AMDGPU backend is not available, tests will not be built")
elseif(DEFINED LLVM_TARGETS_TO_BUILD AND LIBC_TARGET_ARCHITECTURE_IS_AMDGPU
       AND NOT "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(LIBC_GPU_TESTS_DISABLED TRUE)
  message(STATUS "NVPTX backend is not available, tests will not be built")
elseif(LIBC_GPU_TEST_ARCHITECTURE)
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

# The NVPTX backend cannot currently handle objects created in debug mode.
if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(LIBC_GPU_TESTS_DISABLED TRUE)
endif()

# Identify the GPU loader utility used to run tests.
set(LIBC_GPU_LOADER_EXECUTABLE "" CACHE STRING "Executable for the GPU loader.")
if(LIBC_GPU_LOADER_EXECUTABLE)
  set(gpu_loader_executable ${LIBC_GPU_LOADER_EXECUTABLE})
elseif(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  find_program(LIBC_AMDHSA_LOADER_EXECUTABLE
               NAMES amdhsa-loader NO_DEFAULT_PATH
               PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  if(LIBC_AMDHSA_LOADER_EXECUTABLE)
    set(gpu_loader_executable ${LIBC_AMDHSA_LOADER_EXECUTABLE})
  endif()
elseif(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  find_program(LIBC_NVPTX_LOADER_EXECUTABLE
               NAMES nvptx-loader NO_DEFAULT_PATH
               PATHS ${LLVM_BINARY_DIR}/bin ${compiler_path})
  if(LIBC_NVPTX_LOADER_EXECUTABLE)
    set(gpu_loader_executable ${LIBC_NVPTX_LOADER_EXECUTABLE})
  endif()
endif()
if(NOT TARGET libc.utils.gpu.loader AND gpu_loader_executable)
  add_custom_target(libc.utils.gpu.loader)
  set_target_properties(
    libc.utils.gpu.loader
    PROPERTIES
      EXECUTABLE "${gpu_loader_executable}"
  )
endif()

if(LIBC_TARGET_ARCHITECTURE_IS_AMDGPU)
  # The AMDGPU environment uses different code objects to encode the ABI for
  # kernel calls and intrinsic functions. We want to specify this manually to
  # conform to whatever the test suite was built to handle.
  set(LIBC_GPU_CODE_OBJECT_VERSION 5)
endif()

if(LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  # FIXME: This is a hack required to keep the CUDA package from trying to find
  #        pthreads. We only link the CUDA driver, so this is unneeded.
  add_library(CUDA::cudart_static_deps IMPORTED INTERFACE)

  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    get_filename_component(LIBC_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY ABSOLUTE)
  endif()
endif()
