# CMake policy settings shared between LLVM projects

# CMP0182: Create shared library archives by default on AIX.
# New in CMake 4.0: https://cmake.org/cmake/help/latest/policy/CMP0182.html
if(POLICY CMP0182)
  cmake_policy(SET CMP0182 NEW)
endif()
