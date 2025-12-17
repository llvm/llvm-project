cmake_minimum_required(VERSION 3.20.0)
project(RippleSMEVectorLibrary C CXX)

# Set target architecture
set(COMPILER_RT_DEFAULT_TARGET_ARCH aarch64)
set(RIPPLE_SUPPORTED_ARCH aarch64)

# Check if target is supported
if(NOT "${COMPILER_RT_DEFAULT_TARGET_ARCH}" IN_LIST RIPPLE_SUPPORTED_ARCH)
  message(WARNING "Target ${COMPILER_RT_DEFAULT_TARGET_ARCH} not supported.")
  return()
endif()

set(RIPPLE_RT_HEADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/header)

# Source file
set(SRC_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/sme_veclib.c)

# Output directory
set(BC_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/bitcode)

# Output bitcode file
set(BC_FILE ${BC_BUILD_DIR}/sme_veclib.bc)

# Validate compiler supports LLVM bitcode generation
if(NOT CMAKE_C_COMPILER MATCHES "clang")
  message(WARNING "CMAKE_C_COMPILER is not clang, LLVM bitcode generation may fail")
endif()

# Check source file exists
if(NOT EXISTS ${SRC_FILE})
  message(FATAL_ERROR "Source file ${SRC_FILE} not found")
endif()

# Compile to LLVM bitcode
add_custom_command(
  OUTPUT ${BC_FILE}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BC_BUILD_DIR}
  COMMAND ${CMAKE_C_COMPILER}
          --target=aarch64-linux-gnu
	  -march=armv9.2-a+sme
          -O2
	  -msve-streaming-vector-bits=512
          -emit-llvm
          -c ${SRC_FILE}
          -o ${BC_FILE}
  DEPENDS ${SRC_FILE}
  COMMENT "Compiling sme_veclib.c to LLVM bitcode for AArch64"
  VERBATIM
)

# Add target
add_custom_target(build_bitcode ALL DEPENDS ${BC_FILE})

# Install bitcode
install(FILES ${BC_FILE}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/linux/ripple/
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

# Install header
file(GLOB RIPPLE_RT_HEADERS ${RIPPLE_RT_HEADER_DIR}/ripple_sme_veclib.h)
install(
  FILES ${RIPPLE_RT_HEADERS}
  COMPONENT compiler-rt-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/
)
