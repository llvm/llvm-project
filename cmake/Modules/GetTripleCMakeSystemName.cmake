#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Map a target triple to the corresponding CMake system name.
#
# Usage:
#   get_triple_cmake_system_name(<triple> <out_var>)
#
# Sets <out_var> to the CMake-style system name
# (e.g. "x86_64-pc-linux-gnu" -> Linux, "arm64-apple-macos" ->
# "Darwin"). A triple whose OS CMake does not recognize maps to
# "Generic"; a triple with no identifiable OS falls back to
# CMAKE_HOST_SYSTEM_NAME.
#
# The recognized OS and environment names mirror the tables in
# llvm/include/llvm/TargetParser/TripleName.def (and the classification
# mirrors Triple::normalize), but the OS -> CMake system name mapping is
# maintained here: the CMake system names are a property of CMake, not of
# llvm::Triple, and can vary by CMake version.
#
# llvm/test/tools/TargetParser/get-triple-cmake-system-name.cmake unit
# tests this module and guards against the .def OS/environment tables
# drifting out of sync with the mapping below.

# Populate the OS/environment name tables in the caller's scope. Defined as
# a macro so both the classifier and the unit test share one copy of the data.
#
# Order is significant: matching is by prefix, so a name that is a prefix of
# another must come first (wasip1/2/3 before wasi). _GTCSN_OS_NAMES and
# _GTCSN_OS_CMAKE are parallel lists. Only environments that force a system
# name regardless of the OS appear in the environment tables.
macro(_gtcsn_populate_tables)
  set(_GTCSN_OS_NAMES
      darwin dragonfly freebsd fuchsia ios kfreebsd linux lv2 macosx macos
      managarm netbsd openbsd solaris uefi windows win32 zos haiku rtems aix
      cuda nvcl amdhsa ps4 ps5 elfiamcu tvos watchos bridgeos driverkit xros
      visionos mesa3d amdpal hermit hurd wasip1 wasip2 wasip3 wasi emscripten
      shadermodel liteos serenity vulkan cheriotrtos opencl chipstar firmware
      qurt h2)
  set(_GTCSN_OS_CMAKE
      Darwin DragonFly FreeBSD Fuchsia iOS FreeBSD Linux Generic Darwin Darwin
      Generic NetBSD OpenBSD SunOS Generic Windows Windows OS390 Haiku RTEMS AIX
      Generic Generic Generic Generic Generic Generic tvOS watchOS Darwin Darwin
      visionOS visionOS Generic Generic Generic GNU WASI WASI WASI WASI
      Emscripten Generic Generic SerenityOS Generic Generic Generic Generic
      Generic Generic Generic)

  set(_GTCSN_ENV_NAMES android cygnus)
  set(_GTCSN_ENV_OVERRIDE Android CYGWIN)
endmacro()

# Return TRUE in result_var if str starts with prefix.
function(_gtcsn_starts_with str prefix result_var)
  string(FIND "${str}" "${prefix}" _pos)
  if(_pos EQUAL 0)
    set(${result_var} TRUE PARENT_SCOPE)
  else()
    set(${result_var} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Classify a triple to a CMake system name, or the empty string if no OS
# component can be identified (the caller then falls back to the host system
# name). This mirrors the component classification of Triple::normalize: it
# does not trust fixed positions, since a triple may omit the vendor
# (aarch64-linux-android), put the OS in the vendor slot (wasm32-wasi,
# i686-linux), or be a bare OS with no arch (clang accepts --target=darwin as
# unknown-unknown-darwin). No arch name is a prefix of any OS/env name, so
# scanning the first component too cannot misclassify a real arch.
function(_get_triple_cmake_system_name_raw triple out_var)
  _gtcsn_populate_tables()

  string(REPLACE "-" ";" _parts "${triple}")
  list(LENGTH _parts _nparts)

  # An environment override takes precedence over the OS mapping (e.g. android
  # wins over the linux in aarch64-linux-android), so scan for it first.
  list(LENGTH _GTCSN_ENV_NAMES _nenv)
  if(_nenv GREATER 0)
    math(EXPR _last_env "${_nenv} - 1")
    foreach(_comp IN LISTS _parts)
      foreach(_i RANGE ${_last_env})
        list(GET _GTCSN_ENV_NAMES ${_i} _name)
        _gtcsn_starts_with("${_comp}" "${_name}" _match)
        if(_match)
          list(GET _GTCSN_ENV_OVERRIDE ${_i} _override)
          set(${out_var} "${_override}" PARENT_SCOPE)
          return()
        endif()
      endforeach()
    endforeach()
  endif()

  # Find the first component that classifies as an OS. A specific (non-Generic)
  # name wins immediately. A "Generic" match means the component is a recognized
  # OS with no CMake equivalent; stop scanning once any OS is identified.
  list(LENGTH _GTCSN_OS_NAMES _nos)
  math(EXPR _last_os "${_nos} - 1")
  set(_found_generic_os FALSE)
  foreach(_comp IN LISTS _parts)
    set(_matched FALSE)
    foreach(_i RANGE ${_last_os})
      list(GET _GTCSN_OS_NAMES ${_i} _name)
      _gtcsn_starts_with("${_comp}" "${_name}" _match)
      if(_match)
        list(GET _GTCSN_OS_CMAKE ${_i} _cmake)
        if(NOT _cmake STREQUAL "Generic")
          set(${out_var} "${_cmake}" PARENT_SCOPE)
          return()
        endif()
        set(_found_generic_os TRUE)
        set(_matched TRUE)
        break()
      endif()
    endforeach()
    if(_matched)
      break()
    endif()
  endforeach()

  # Legacy OS spellings that normalize rewrites to windows.
  foreach(_comp IN LISTS _parts)
    _gtcsn_starts_with("${_comp}" "mingw" _match)
    if(_match)
      set(${out_var} "Windows" PARENT_SCOPE)
      return()
    endif()
    _gtcsn_starts_with("${_comp}" "cygwin" _match_cygwin)
    _gtcsn_starts_with("${_comp}" "msys" _match_msys)
    if(_match_cygwin OR _match_msys)
      set(${out_var} "CYGWIN" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  # Apple vendor triples with no specific CMake OS name (e.g. firmware) map to
  # Darwin.
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

  # No OS could be identified. Too few components to have one -> let the caller
  # fall back to the host system name.
  if(_nparts LESS 3)
    set(${out_var} "" PARENT_SCOPE)
  else()
    set(${out_var} "Generic" PARENT_SCOPE)
  endif()
endfunction()

function(get_triple_cmake_system_name triple out_var)
  _get_triple_cmake_system_name_raw("${triple}" _name)
  if(_name)
    set(${out_var} "${_name}" PARENT_SCOPE)
  else()
    set(${out_var} "${CMAKE_HOST_SYSTEM_NAME}" PARENT_SCOPE)
  endif()
endfunction()
