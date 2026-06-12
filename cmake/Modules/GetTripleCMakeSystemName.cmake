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
# CMake-style system name (e.g. "Darwin", "Linux", "Windows").
# Unrecognized OS values are mapped to "Generic".

function(get_triple_cmake_system_name triple out_var)
  string(REPLACE "-" ";" _components "${triple}")
  list(LENGTH _components _len)
  if(_len LESS 3)
    set(${out_var} "${CMAKE_HOST_SYSTEM_NAME}" PARENT_SCOPE)
    return()
  endif()

  list(GET _components 1 _vendor)
  list(GET _components 2 _os)
  set(_env "")
  if(_len GREATER_EQUAL 4)
    list(GET _components 3 _env)
  endif()

  # Check the special environment components first, since it can
  # override the usual OS mapping.
  if("${_env}" MATCHES "^android")
    set(${out_var} "Android" PARENT_SCOPE)
  elseif("${_env}" MATCHES "^cygnus|^cygwin")
    set(${out_var} "CYGWIN" PARENT_SCOPE)
  elseif("${_env}" MATCHES "^msys")
    set(${out_var} "MSYS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^darwin|^macos")
    set(${out_var} "Darwin" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^ios")
    set(${out_var} "iOS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^tvos")
    set(${out_var} "tvOS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^watchos")
    set(${out_var} "watchOS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^xros|^visionos")
    set(${out_var} "visionOS" PARENT_SCOPE)
  elseif("${_vendor}" STREQUAL "apple")
    # Catch-all for other Apple triples (e.g. driverkit, bridgeos).
    set(${out_var} "Darwin" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^linux")
    set(${out_var} "Linux" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^win32|^windows|^mingw")
    set(${out_var} "Windows" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^freebsd|^kfreebsd")
    set(${out_var} "FreeBSD" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^netbsd")
    set(${out_var} "NetBSD" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^openbsd")
    set(${out_var} "OpenBSD" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^dragonfly")
    set(${out_var} "DragonFly" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^solaris")
    set(${out_var} "SunOS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^aix")
    set(${out_var} "AIX" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^fuchsia")
    set(${out_var} "Fuchsia" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^haiku")
    set(${out_var} "Haiku" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^emscripten")
    set(${out_var} "Emscripten" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^wasi")
    set(${out_var} "WASI" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^rtems")
    set(${out_var} "RTEMS" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^zos")
    set(${out_var} "OS390" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^hurd")
    set(${out_var} "GNU" PARENT_SCOPE)
  elseif("${_os}" MATCHES "^serenity")
    set(${out_var} "SerenityOS" PARENT_SCOPE)
  else()
    set(${out_var} "Generic" PARENT_SCOPE)
  endif()
endfunction()
