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

# Build FortranDecimal when the build target is Flang or LLVM.
if (CMAKE_SOURCE_DIR STREQUAL FLANG_SOURCE_DIR OR CMAKE_SOURCE_DIR STREQUAL LLVM_MAIN_SRC_DIR)
  add_flang_library(FortranDecimal
    binary-to-decimal.cpp
    decimal-to-binary.cpp
  )
# Build FortranDecimalRT for FlangRT when the build target is Runtimes.
# Standalone builds of FlangRT is not supported.
elseif (CMAKE_SOURCE_DIR STREQUAL Runtimes_SOURCE_DIR)
  add_flang_library(FortranDecimalRT STATIC INSTALL_WITH_TOOLCHAIN PIC
    binary-to-decimal.cpp
    decimal-to-binary.cpp
  )
else()
  message(FATAL_ERROR "CMAKE_SOURCE_DIR of target points to neither Flang or Flang-rt, no library added for FortranDecimal.")
endif()
