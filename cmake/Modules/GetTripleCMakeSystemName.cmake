#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

# Extract the OS component from a target triple and map it to the
# corresponding CMake system name.
#
# Usage:
#   get_triple_cmake_system_name(<triple> <out_var>)
#
# Parses the triple (arch-vendor-os[-env]) and sets <out_var> to the
# CMake-style system name (e.g. "Darwin", "Linux",
# "Windows"). Unrecognized OS values are mapped to "Generic". This
# makes some attempt to tolerate unnormalized triples.
function(get_triple_cmake_system_name triple out_var)
  string(REPLACE "-" ";" _components "${triple}")
  list(LENGTH _components _len)
  if(_len LESS 2)
    set(${out_var} "${CMAKE_HOST_SYSTEM_NAME}" PARENT_SCOPE)
    return()
  endif()

  # The environment tokens take precedence over the OS mapping (e.g. an
  # "android" or "cygnus" environment on top of a "linux"/"windows" OS
  # determines the CMake system name).
  foreach(_c IN LISTS _components)
    if("${_c}" MATCHES "^android")
      set(${out_var} "Android" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^cygnus|^cygwin")
      # "cygnus" is the environment in a normalized triple
      # (e.g. x86_64-pc-windows-cygnus); "cygwin" is the OS component in the
      # un-normalized form (e.g. x86_64-pc-cygwin).
      set(${out_var} "CYGWIN" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  # Apple triples may use a bare vendor ("apple") with an OS that isn't matched
  # below (e.g. "driverkit", "bridgeos"); treat any apple-vendored triple as
  # Darwin as a catch-all after the specific OS checks.
  set(_apple_vendor OFF)

  foreach(_c IN LISTS _components)
    if("${_c}" MATCHES "^darwin|^macos")
      set(${out_var} "Darwin" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^ios")
      set(${out_var} "iOS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^tvos")
      set(${out_var} "tvOS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^watchos")
      set(${out_var} "watchOS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^xros|^visionos")
      set(${out_var} "visionOS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^linux")
      set(${out_var} "Linux" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^windows")
      set(${out_var} "Windows" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^freebsd|^kfreebsd")
      set(${out_var} "FreeBSD" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^netbsd")
      set(${out_var} "NetBSD" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^openbsd")
      set(${out_var} "OpenBSD" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^dragonfly")
      set(${out_var} "DragonFly" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^solaris")
      set(${out_var} "SunOS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^aix")
      set(${out_var} "AIX" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^fuchsia")
      set(${out_var} "Fuchsia" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^haiku")
      set(${out_var} "Haiku" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^emscripten")
      set(${out_var} "Emscripten" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^wasi")
      set(${out_var} "WASI" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^rtems")
      set(${out_var} "RTEMS" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^zos")
      set(${out_var} "OS390" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^hurd")
      set(${out_var} "GNU" PARENT_SCOPE)
      return()
    elseif("${_c}" MATCHES "^serenity")
      set(${out_var} "SerenityOS" PARENT_SCOPE)
      return()
    elseif("${_c}" STREQUAL "apple")
      set(_apple_vendor ON)
    endif()
  endforeach()

  if(_apple_vendor)
    # Catch-all for other Apple triples (e.g. driverkit, bridgeos).
    set(${out_var} "Darwin" PARENT_SCOPE)
    return()
  endif()

  set(${out_var} "Generic" PARENT_SCOPE)
endfunction()
