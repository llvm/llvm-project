# Install script for directory: /home/angandhi/llvm-project/amd/device-libs/oclc

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/angandhi/llvm-project/amd/device-libs/build/dist")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_400.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_500.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_abi_version_600.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_correctly_rounded_sqrt_off.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_daz_opt_off.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_daz_opt_on.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_finite_only_off.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_finite_only_on.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_10-1-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_10-3-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1010.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1011.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1012.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1013.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1030.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1031.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1032.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1033.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1034.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1035.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1036.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_11-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1100.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1101.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1102.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1103.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1150.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1151.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1152.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1153.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_12-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1200.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_1201.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_600.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_601.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_602.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_700.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_701.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_702.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_703.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_704.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_705.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_801.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_802.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_803.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_805.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_810.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_9-4-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_9-generic.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_900.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_902.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_904.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_906.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_908.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_909.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_90a.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_90c.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_942.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_isa_version_950.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_unsafe_math_off.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_unsafe_math_on.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_wavefrontsize64_off.bc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "device-libs" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/amdgcn/bitcode" TYPE FILE FILES "/home/angandhi/llvm-project/amd/device-libs/build/amdgcn/bitcode/oclc_wavefrontsize64_on.bc")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/angandhi/llvm-project/amd/device-libs/build/oclc/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
