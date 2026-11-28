#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# get_triple_cmake_system_name(<triple> <out_var>)
#
# Set <out_var> to the CMAKE_SYSTEM_NAME for a target triple, e.g. Linux for
# x86_64-pc-linux-gnu. OSes without a CMake equivalent map to Generic. If the
# triple has no OS, falls back to CMAKE_HOST_SYSTEM_NAME.
#
# The OS and environment names must match TripleName.def, which is checked by
# llvm/test/CMake/get-triple-cmake-system-name.test.

# Matching is by prefix, so a name must come before any longer name it is a
# prefix of (e.g. wasip1 before wasi). Only environments that determine the
# system name are listed.
macro(_gtcsn_populate_tables)
  set(_gtcsn_os_map
      darwin=Darwin
      dragonfly=DragonFly
      freebsd=FreeBSD
      fuchsia=Fuchsia
      ios=iOS
      kfreebsd=FreeBSD
      linux=Linux
      lv2=Generic
      macosx=Darwin
      macos=Darwin
      managarm=Generic
      netbsd=NetBSD
      openbsd=OpenBSD
      solaris=SunOS
      uefi=Generic
      windows=Windows
      win32=Windows
      zos=OS390
      haiku=Haiku
      rtems=RTEMS
      aix=AIX
      cuda=Generic
      nvcl=Generic
      amdhsa=Generic
      ps4=Generic
      ps5=Generic
      elfiamcu=Generic
      tvos=tvOS
      watchos=watchOS
      bridgeos=Darwin
      driverkit=Darwin
      xros=visionOS
      visionos=visionOS
      mesa3d=Generic
      amdpal=Generic
      hermit=Generic
      hurd=GNU
      wasip1=WASI
      wasip2=WASI
      wasip3=WASI
      wasi=WASI
      emscripten=Emscripten
      shadermodel=Generic
      liteos=Generic
      serenity=SerenityOS
      vulkan=Generic
      cheriotrtos=Generic
      opencl=Generic
      chipstar=Generic
      firmware=Generic
      qurt=Generic
      h2=Generic)

  set(_gtcsn_env_map
      android=Android
      cygnus=CYGWIN)

  foreach(_gtcsn_kind OS ENV)
    string(TOLOWER "${_gtcsn_kind}" _gtcsn_lower)
    set(_GTCSN_${_gtcsn_kind}_NAMES "")
    set(_GTCSN_${_gtcsn_kind}_CMAKE "")
    foreach(_gtcsn_entry IN LISTS _gtcsn_${_gtcsn_lower}_map)
      string(REPLACE "=" ";" _gtcsn_entry "${_gtcsn_entry}")
      list(GET _gtcsn_entry 0 _gtcsn_name)
      list(GET _gtcsn_entry 1 _gtcsn_cmake)
      list(APPEND _GTCSN_${_gtcsn_kind}_NAMES "${_gtcsn_name}")
      list(APPEND _GTCSN_${_gtcsn_kind}_CMAKE "${_gtcsn_cmake}")
    endforeach()
  endforeach()
endmacro()

# Returns an empty string if no OS is found. Like Triple::normalize, this checks
# every component rather than relying on fixed positions, so unnormalized
# triples such as aarch64-linux-android, wasm32-wasi and darwin also work.
function(_get_triple_cmake_system_name_raw triple out_var)
  _gtcsn_populate_tables()

  string(REPLACE "-" ";" _parts "${triple}")
  list(LENGTH _parts _nparts)

  # The environment takes precedence, e.g. aarch64-linux-android is
  # Android, not Linux.
  foreach(_comp IN LISTS _parts)
    foreach(_name _override IN ZIP_LISTS _GTCSN_ENV_NAMES _GTCSN_ENV_CMAKE)
      if(_comp MATCHES "^${_name}")
        set(${out_var} "${_override}" PARENT_SCOPE)
        return()
      endif()
    endforeach()
  endforeach()

  # Stop at the first component that is a known OS.
  set(_found_generic_os FALSE)
  foreach(_comp IN LISTS _parts)
    foreach(_name _cmake IN ZIP_LISTS _GTCSN_OS_NAMES _GTCSN_OS_CMAKE)
      if(_comp MATCHES "^${_name}")
        if(NOT _cmake STREQUAL "Generic")
          set(${out_var} "${_cmake}" PARENT_SCOPE)
          return()
        endif()
        set(_found_generic_os TRUE)
        break()
      endif()
    endforeach()

    if(_found_generic_os)
      break()
    endif()
  endforeach()

  # Legacy spellings that Triple::normalize turns into windows.
  foreach(_comp IN LISTS _parts)
    if(_comp MATCHES "^mingw")
      set(${out_var} "Windows" PARENT_SCOPE)
      return()
    endif()

    if(_comp MATCHES "^(cygwin|msys)")
      set(${out_var} "CYGWIN" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  # Apple OSes without a more specific name, e.g. firmware.
  if(_nparts GREATER 1)
    list(GET _parts 1 _vendor)
    if(_vendor STREQUAL "apple")
      set(${out_var} "Darwin" PARENT_SCOPE)
      return()
    endif()
  endif()

  if(_found_generic_os)
    set(${out_var} "Generic" PARENT_SCOPE)
    return()
  endif()

  # A triple with fewer than three components may just be missing the OS.
  if(_nparts LESS 3)
    set(${out_var} "" PARENT_SCOPE)
  else()
    set(${out_var} "Generic" PARENT_SCOPE)
  endif()
endfunction()

function(get_triple_cmake_system_name triple out_var)
  _get_triple_cmake_system_name_raw("${triple}" _name)
  if(NOT _name STREQUAL "")
    set(${out_var} "${_name}" PARENT_SCOPE)
  else()
    set(${out_var} "${CMAKE_HOST_SYSTEM_NAME}" PARENT_SCOPE)
  endif()
endfunction()
