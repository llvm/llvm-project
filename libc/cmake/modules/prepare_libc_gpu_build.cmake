if(NOT LIBC_TARGET_ARCHITECTURE_IS_GPU)
  message(FATAL_ERROR
          "libc build: Invalid attempt to set up GPU architectures.")
endif()

# Set up the target architectures to build the GPU libc for.
set(all_gpu_architectures "sm_35;sm_37;sm_50;sm_52;sm_53;sm_60;sm_61;sm_62;"
                          "sm_70;sm_72;sm_75;sm_80;sm_86;gfx700;gfx701;gfx801;"
                          "gfx803;gfx900;gfx902;gfx906;gfx908;gfx90a;gfx90c;"
                          "gfx940;gfx1010;gfx1030;gfx1031;gfx1032;gfx1033;"
                          "gfx1034;gfx1035;gfx1036;gfx1100;gfx1101;gfx1102;"
                          "gfx1103")
set(LIBC_GPU_ARCHITECTURES ${all_gpu_architectures} CACHE STRING
    "List of GPU architectures to build the libc for.")
if(LIBC_GPU_ARCHITECTURES STREQUAL "all")
  set(LIBC_GPU_ARCHITECTURES ${all_gpu_architectures} FORCE)
endif()

# Ensure the compiler is a valid clang when building the GPU target.
set(req_ver "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang" AND
        ${CMAKE_CXX_COMPILER_VERSION} VERSION_EQUAL "${req_ver}"))
  message(FATAL_ERROR "Cannot build libc for GPU. CMake compiler "
                      "'${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}' "
                      " is not `Clang ${req_ver}.")
endif()
if(NOT LLVM_LIBC_FULL_BUILD)
  message(FATAL_ERROR "LLVM_LIBC_FULL_BUILD must be enabled to build libc for "
                      "GPU.")
endif()
