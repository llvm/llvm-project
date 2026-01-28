#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

if(NOT TARGET llvm-libc-common-utilities)
  set(libc_path ${CMAKE_CURRENT_LIST_DIR}/../../libc)
  if (EXISTS ${libc_path} AND IS_DIRECTORY ${libc_path})
    add_library(llvm-libc-common-utilities INTERFACE)
    # TODO: Reorganize the libc shared section so that it can be included without
    # adding the root "libc" directory to the include path.
    if (NOT(LIBCXX_ENABLE_THREADS))
      target_compile_definitions(llvm-libc-common-utilities INTERFACE LIBC_THREAD_MODE=LIBC_THREAD_MODE_SINGLE)
    endif()
    target_include_directories(llvm-libc-common-utilities INTERFACE ${libc_path})
    target_compile_definitions(llvm-libc-common-utilities INTERFACE LIBC_NAMESPACE=__llvm_libc_common_utils)
    target_compile_features(llvm-libc-common-utilities INTERFACE cxx_std_17)
  endif()
endif()
