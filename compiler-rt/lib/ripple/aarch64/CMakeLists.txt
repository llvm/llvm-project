cmake_minimum_required(VERSION 3.20.0)
project(RippleSMEVectorLibrary C CXX)

set(COMPILER_RT_DEFAULT_TARGET_TRIPLE
    "${COMPILER_RT_DEFAULT_TARGET_TRIPLE}"
    CACHE STRING "Target triple (e.g., aarch64-linux-gnu)")
set(RIPPLE_SUPPORTED_ARCH aarch64)

# Set target architecture
set(_derived_arch "")
# Extract first token (arch) from triple and normalize aliases
if(COMPILER_RT_DEFAULT_TARGET_TRIPLE)
  # Split triple by '-': "<arch>-<vendor>-<os>-<abi>"
  string(REPLACE "-" ";" _triple_parts "${COMPILER_RT_DEFAULT_TARGET_TRIPLE}")
  list(GET _triple_parts 0 _arch_raw)
  string(TOLOWER "${_arch_raw}" _arch)

  # Normalize common aliases to canonical names
  if(_arch STREQUAL "arm64")      # Apple alias for aarch64
    set(_arch "aarch64")
  elseif(_arch STREQUAL "armv8")  # Often equivalent to aarch64
    set(_arch "aarch64")
  endif()

  set(_derived_arch "${_arch}")
endif()

# Check if target is supported
if(NOT _derived_arch)
  message(WARNING
    "Ripple aarch64 vector library disabled: no target triple provided. "
    "Pass -DCOMPILER_RT_DEFAULT_TARGET_TRIPLE=aarch64-linux-gnu to enable.")
  return()
elseif(NOT "${_derived_arch}" IN_LIST RIPPLE_SUPPORTED_ARCH)
  message(WARNING
    "Ripple aarch64 vector library disabled. "
    "Target ${_derived_arch} not supported.")
  return()
else()
  message(STATUS "Ripple vector library enabled for ${_derived_arch}")
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
