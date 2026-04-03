# Determine the subdirectory relative to Clang's resource dir/installation
# prefix where to install target-specific libraries, to be found by the
# Clang/Flang driver. This was adapted from Compiler-RT's mechanism to find the
# path for libclang_rt.builtins.a.
#
# Compiler-RT has two mechanisms for the path (simplified):
#
# * LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF: lib/${oslibname}/libclang_rt.builtins-${arch}.a
# * LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON : lib/${triple}/libclang_rt.builtins.a
#
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON is the newer scheme, but the old one is
# currently still used by default for some platforms such as Windows. Clang
# looks for which of the files exist before passing the path to the linker.
# Hence, the directories have to match what Clang is looking for, which is done
# in ToolChain::getArchSpecificLibPaths(..), ToolChain::getRuntimePath(),
# ToolChain::getCompilerRTPath(), and ToolChain::getCompilerRT(..), not entirely
# consistent between these functions, overrides in different toolchains, and
# Compiler-RT's CMake code.
#
# For simplicity, we always use the second scheme even as if
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON, even if it actually set to OFF.
# The driver's lookup does not take the build-time
# LLVM_ENABLE_PER_TARGET_RUNTIME_DIR setting into account anyway
# and will always find its files using either scheme.
function (get_toolchain_library_subdir outvar)
  set(outval "lib")

  if (APPLE)
    # Required to be "darwin" for MachO toolchain.
    get_toolchain_os_dirname(os_dirname)
    set(outval "${outval}/${os_dirname}")
  else ()
    get_toolchain_target_dirname(target_dirname)
    set(outval "${outval}/${target_dirname}")
  endif ()

  set(${outvar} "${outval}" PARENT_SCOPE)
endfunction ()


# Corresponds to Flang's ToolChain::getDefaultIntrinsicModuleDir().
function (get_toolchain_fortran_module_subdir outvar)
  set(outval "finclude/flang")

  get_toolchain_target_dirname(target_dirname)
  set(outval "${outval}/${target_dirname}")

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


# Internal function extracted from compiler-rt. Use get_toolchain_target_dirname
# instead for new code.
function(get_runtimes_target_libdir_common default_target_triple arch variable)
  string(FIND "${default_target_triple}" "-" dash_index)
  string(SUBSTRING "${default_target_triple}" "${dash_index}" -1 triple_suffix)
  string(SUBSTRING "${default_target_triple}" 0 "${dash_index}" triple_cpu)
  if(ANDROID AND "${arch}" STREQUAL "i386")
    set(target "i686${triple_suffix}")
  elseif("${arch}" STREQUAL "amd64")
    set(target "x86_64${triple_suffix}")
  elseif("${arch}" STREQUAL "sparc64")
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
    # Arch is arm, armhf, armv6m (anything else would come from using
    # COMPILER_RT_DEFAULT_TARGET_ONLY, which is checked above).
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
  set("${variable}" "${target}" PARENT_SCOPE)
endfunction()


# Corresponds to Clang's ToolChain::getRuntimePath().
function (get_toolchain_target_dirname outvar)
  string(FIND "${LLVM_TARGET_TRIPLE}" "-" dash_index)
  if (dash_index EQUAL "-1")
    # This means LLVM_TARGET_TRIPLE is not set and we cannot derive the dirname
    # from it. The proper behavior here would be to emit an error since we have
    # no target we can build for. However, compiler-rt uses
    # COMPILER_RT_DEFAULT_TARGET_TRIPLE instead and ignores LLVM_TARGET_TRIPLE.
    # To not break the build when building only compiler-rt, we skip the triple
    # subdirectory.
    set(target "")
  else ()
    string(SUBSTRING "${LLVM_TARGET_TRIPLE}" 0 "${dash_index}" triple_cpu)
    set(arch "${triple_cpu}")
    if("${arch}" MATCHES "^i.86$")
      set(arch "i386")
    endif()
    get_runtimes_target_libdir_common("${LLVM_TARGET_TRIPLE}" "${arch}" target)
  endif ()
  set("${outvar}" "${target}" PARENT_SCOPE)
endfunction()
