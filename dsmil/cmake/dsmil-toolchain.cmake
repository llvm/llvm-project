# DSMIL toolchain preset for consumers of an installed DSLLVM.
#
# Usage:
#   cmake -G Ninja -S <src> -B <build> \
#     -DCMAKE_TOOLCHAIN_FILE=/usr/local/share/dsmil/dsmil-toolchain.cmake
#
# Customize the install prefix if DSLLVM was installed somewhere else:
#   cmake ... -DDSMIL_PREFIX=/opt/dsllvm ...

set(DSMIL_PREFIX "/usr/local" CACHE PATH "Install prefix of DSLLVM/DSMIL toolchain")

set(_DSMIL_BIN "${DSMIL_PREFIX}/bin")
set(_DSMIL_LIB "${DSMIL_PREFIX}/lib")
set(_DSMIL_INC "${DSMIL_PREFIX}/include/dsmil")

set(CMAKE_C_COMPILER "${_DSMIL_BIN}/dsmil-clang" CACHE FILEPATH "DSMIL C compiler" FORCE)
set(CMAKE_CXX_COMPILER "${_DSMIL_BIN}/dsmil-clang++" CACHE FILEPATH "DSMIL C++ compiler" FORCE)

set(CMAKE_PREFIX_PATH "${DSMIL_PREFIX}" CACHE PATH "Search prefix for DSMIL libraries" FORCE)

# Ensure CMake can find the runtime and plugin from the install.
set(CMAKE_LIBRARY_PATH "${_DSMIL_LIB}" CACHE PATH "DSMIL library path" FORCE)
set(CMAKE_INCLUDE_PATH "${_DSMIL_INC}" CACHE PATH "DSMIL include path" FORCE)

if(NOT EXISTS "${CMAKE_C_COMPILER}")
  message(FATAL_ERROR "dsmil-clang not found at ${CMAKE_C_COMPILER}. Set DSMIL_PREFIX to your install.")
endif()
if(NOT EXISTS "${CMAKE_CXX_COMPILER}")
  message(FATAL_ERROR "dsmil-clang++ not found at ${CMAKE_CXX_COMPILER}. Set DSMIL_PREFIX to your install.")
endif()
