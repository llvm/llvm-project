if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.20.0)

  project(FortranDecimal C CXX)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)

  set(FLANG_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

  set(LLVM_COMMON_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../cmake")
  set(LLVM_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../llvm/cmake")
  set(CLANG_CMAKE_UTILS "${FLANG_SOURCE_DIR}/../clang/cmake")

  # Add path for custom modules
  list(INSERT CMAKE_MODULE_PATH 0
    "${FLANG_SOURCE_DIR}/cmake"
    "${FLANG_SOURCE_DIR}/cmake/modules"
    "${LLVM_COMMON_CMAKE_UTILS}"
    "${LLVM_COMMON_CMAKE_UTILS}/Modules"
    "${LLVM_CMAKE_UTILS}"
    "${LLVM_CMAKE_UTILS}/modules"
    "${CLANG_CMAKE_UTILS}/modules"
    )

  include(AddClang)
  include(AddLLVM)
  include(AddFlang)
  include(HandleLLVMOptions)

  include(TestBigEndian)
  test_big_endian(IS_BIGENDIAN)
  if (IS_BIGENDIAN)
    add_compile_definitions(FLANG_BIG_ENDIAN=1)
  else ()
    add_compile_definitions(FLANG_LITTLE_ENDIAN=1)
  endif ()
  include_directories(BEFORE
    ${FLANG_SOURCE_DIR}/include)
endif()

check_cxx_compiler_flag(-fno-lto FLANG_RUNTIME_HAS_FNO_LTO_FLAG)
if (FLANG_RUNTIME_HAS_FNO_LTO_FLAG)
  append("-fno-lto" CMAKE_CXX_FLAGS)
endif()

# Disable libstdc++ assertions, even in an LLVM_ENABLE_ASSERTIONS build, to
# avoid an unwanted dependency on libstdc++.so.
add_definitions(-U_GLIBCXX_ASSERTIONS)

set(sources
  binary-to-decimal.cpp
  decimal-to-binary.cpp
)

include(AddFlangOffloadRuntime)
enable_cuda_compilation(FortranDecimal "${sources}")
enable_omp_offload_compilation("${sources}")

add_flang_library(FortranDecimal INSTALL_WITH_TOOLCHAIN ${sources})

if (DEFINED MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)
  add_flang_library(FortranDecimal.static INSTALL_WITH_TOOLCHAIN
          binary-to-decimal.cpp
          decimal-to-binary.cpp
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
  add_flang_library(FortranDecimal.dynamic INSTALL_WITH_TOOLCHAIN
    binary-to-decimal.cpp
    decimal-to-binary.cpp
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebug)
  add_flang_library(FortranDecimal.static_dbg INSTALL_WITH_TOOLCHAIN
          binary-to-decimal.cpp
          decimal-to-binary.cpp
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDebugDLL)
  add_flang_library(FortranDecimal.dynamic_dbg INSTALL_WITH_TOOLCHAIN
    binary-to-decimal.cpp
    decimal-to-binary.cpp
  )
  add_dependencies(FortranDecimal FortranDecimal.static FortranDecimal.dynamic
    FortranDecimal.static_dbg FortranDecimal.dynamic_dbg)
endif()
