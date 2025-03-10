#===-- cmake/modules/GetToolchainDirs.cmake --------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#


# Determine the subdirectory relative to Clang's resource dir/sysroot where to
# install target-specific libraries, to be found by Clang/Flang driver. This was
# adapted from Compiler-RT's mechanism to find the path for
# libclang_rt.builtins.a.
#
# Compiler-RT has two mechanisms for the path (simplified):
#
# * LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=1: lib/${oslibname}/libclang_rt.builtins-${arch}.a
# * LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=0: lib/${triple}/libclang_rt.builtins.a
#
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON is the newer scheme, but the old one is
# currently still used for some platforms such as Windows. Clang looks for which
# of the files exist before passing the path to the linker. Hence, the
# directories have to match what Clang is looking for, which is done in
# ToolChain::getArchSpecificLibPaths(..), ToolChain::getRuntimePath(),
# ToolChain::getCompilerRTPath(), and ToolChain::getCompilerRT(..), not entirely
# consistent between these functions, Compiler-RT's CMake code, and overrides
# in different toolchains.
#
# For Fortran, Flang always assumes the library name libflang_rt.a without
# architecture suffix. Hence, we always use the second scheme even as if
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON, even if it actually set to OFF. It as
# added unconditionally to the library search path by
# ToolChain::getArchSpecificLibPaths(...).
function (get_toolchain_library_subdir outvar)
  if (NOT APPLE)
    set(outval "lib")
  else ()
    # Required to be "darwin" for MachO toolchain.
    get_toolchain_os_dirname(os_dirname)
    set(outval "lib/${os_dirname}")
  endif ()

  get_toolchain_arch_dirname(arch_dirname)
  set(outval "lib/${arch_dirname}")

  set(${outvar} "${outval}" PARENT_SCOPE)
endfunction ()


# Corresponds to Clang's ToolChain::getOSLibName(). Adapted from Compiler-RT.
function (get_toolchain_os_dirname outvar)
  if (ANDROID)
    # The CMAKE_SYSTEM_NAME for Android is "Android", but the OS is Linux and the
    # driver will search for libraries in the "linux" directory.
    set(outval "linux")
  else ()
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" outval)
  endif ()
  set(${outvar} "${outval}" PARENT_SCOPE)
endfunction ()


# Corresponds to Clang's ToolChain::getRuntimePath(). Adapted from Compiler-RT.
function (get_toolchain_arch_dirname outvar)
  string(REPLACE "-" ";" triple_list ${LLVM_TARGET_TRIPLE})
  list(GET triple_list 0 arch)

  if("${arch}" MATCHES "^i.86$")
    # Android uses i686, but that's remapped at a later stage.
    set(arch "i386")
  endif()

  string(FIND ${LLVM_TARGET_TRIPLE} "-" dash_index)
  string(SUBSTRING ${LLVM_TARGET_TRIPLE} ${dash_index} -1 triple_suffix)
  string(SUBSTRING ${LLVM_TARGET_TRIPLE} 0 ${dash_index} triple_cpu)
  set(arch "${triple_cpu}")
  if("${arch}" MATCHES "^i.86$")
    # Android uses i686, but that's remapped at a later stage.
    set(arch "i386")
  endif()

  if(ANDROID AND ${arch} STREQUAL "i386")
    set(target "i686${triple_suffix}")
  elseif(${arch} STREQUAL "amd64")
    set(target "x86_64${triple_suffix}")
  elseif(${arch} STREQUAL "sparc64")
    set(target "sparcv9${triple_suffix}")
  elseif("${arch}" MATCHES "mips64|mips64el")
    string(REGEX REPLACE "-gnu.*" "-gnuabi64" triple_suffix_gnu "${triple_suffix}")
    string(REGEX REPLACE "mipsisa32" "mipsisa64" triple_cpu_mips "${triple_cpu}")
    string(REGEX REPLACE "^mips$" "mips64" triple_cpu_mips "${triple_cpu_mips}")
    string(REGEX REPLACE "^mipsel$" "mips64el" triple_cpu_mips "${triple_cpu_mips}")
    set(target "${triple_cpu_mips}${triple_suffix_gnu}")
  elseif("${arch}" MATCHES "mips|mipsel")
    string(REGEX REPLACE "-gnuabi.*" "-gnu" triple_suffix_gnu "${triple_suffix}")
    string(REGEX REPLACE "mipsisa64" "mipsisa32" triple_cpu_mips "${triple_cpu}")
    string(REGEX REPLACE "mips64" "mips" triple_cpu_mips "${triple_cpu_mips}")
    set(target "${triple_cpu_mips}${triple_suffix_gnu}")
  elseif("${arch}" MATCHES "^arm")
    # FIXME: Handle arch other than arm, armhf, armv6m
    if (${arch} STREQUAL "armhf")
      # If we are building for hard float but our ABI is soft float.
      if ("${triple_suffix}" MATCHES ".*eabi$")
        # Change "eabi" -> "eabihf"
        set(triple_suffix "${triple_suffix}hf")
      endif()
      # ABI is already set in the triple, don't repeat it in the architecture.
      set(arch "arm")
    else ()
      # If we are building for soft float, but the triple's ABI is hard float.
      if ("${triple_suffix}" MATCHES ".*eabihf$")
        # Change "eabihf" -> "eabi"
        string(REGEX REPLACE "hf$" "" triple_suffix "${triple_suffix}")
      endif()
    endif()
    set(target "${arch}${triple_suffix}")
  elseif("${arch}" MATCHES "^amdgcn")
    set(target "amdgcn-amd-amdhsa")
  elseif("${arch}" MATCHES "^nvptx")
    set(target "nvptx64-nvidia-cuda")
  else()
    set(target "${arch}${triple_suffix}")
  endif()
  set(${outvar} "${target}" PARENT_SCOPE)
endfunction()
