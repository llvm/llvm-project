#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

set(libc_path ${CMAKE_CURRENT_LIST_DIR}/../../libc)

if(CMAKE_SCRIPT_MODE_FILE)
  if(NOT DEST OR NOT MANIFEST)
    message(FATAL_ERROR "set DEST and MANIFEST")
  endif()
  if(NOT LIBC_ROOT)
    set(LIBC_ROOT ${libc_path})
  endif()

  file(STRINGS "${MANIFEST}" _lines)
  set(_count 0)
  foreach(_rel IN LISTS _lines)
    string(STRIP "${_rel}" _rel)
    if(_rel STREQUAL "" OR _rel MATCHES "^#")
      continue()
    endif()
    if(NOT EXISTS "${LIBC_ROOT}/${_rel}")
      message(FATAL_ERROR "manifest lists a missing file: ${LIBC_ROOT}/${_rel}")
    endif()
    get_filename_component(_dstdir "${DEST}/${_rel}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dstdir}")
    configure_file("${LIBC_ROOT}/${_rel}" "${DEST}/${_rel}" COPYONLY)
    math(EXPR _count "${_count} + 1")
  endforeach()
  message(STATUS "exported ${_count} libc headers into ${DEST}")
  return()
endif()

if(NOT TARGET llvm-libc-common-utilities)
  if (EXISTS ${libc_path} AND IS_DIRECTORY ${libc_path})
    add_library(llvm-libc-common-utilities INTERFACE)
    # TODO: Reorganize the libc shared section so that it can be included without
    # adding the root "libc" directory to the include path.
    if (NOT(LIBCXX_ENABLE_THREADS))
      target_compile_definitions(llvm-libc-common-utilities INTERFACE LIBC_THREAD_MODE=LIBC_THREAD_MODE_SINGLE)
    endif()
    target_include_directories(llvm-libc-common-utilities SYSTEM INTERFACE ${libc_path})
    target_compile_definitions(llvm-libc-common-utilities INTERFACE LIBC_NAMESPACE=__llvm_libc_common_utils)
    target_compile_features(llvm-libc-common-utilities INTERFACE cxx_std_17)
  endif()
endif()
