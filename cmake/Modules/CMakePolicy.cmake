# CMake policy settings shared between LLVM projects

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# MSVC debug information format flags are selected via
# CMAKE_MSVC_DEBUG_INFORMATION_FORMAT, instead of
# embedding flags in e.g. CMAKE_CXX_FLAGS_RELEASE.
# New in CMake 3.25.
#
# Supports debug info with SCCache
# (https://github.com/mozilla/sccache?tab=readme-ov-file#usage)
# avoiding “fatal error C1041: cannot open program database; if
# multiple CL.EXE write to the same .PDB file, please use /FS"
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
endif()

# CMP0144: find_package() uses uppercase <PackageName>_ROOT variables.
# New in CMake 3.27: https://cmake.org/cmake/help/latest/policy/CMP0144.html
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# CMP0147: Visual Studio Generators build custom commands in parallel.
# New in CMake 3.27: https://cmake.org/cmake/help/latest/policy/CMP0147.html
if(POLICY CMP0147)
  cmake_policy(SET CMP0147 NEW)
endif()

# CMP0156: De-duplicate libraries on link lines based on linker capabilities.
# New in CMake 3.29: https://cmake.org/cmake/help/latest/policy/CMP0156.html
# Avoids the deluge of 'ld: warning: ignoring duplicate libraries' warnings when
# building with the Apple linker.
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()
