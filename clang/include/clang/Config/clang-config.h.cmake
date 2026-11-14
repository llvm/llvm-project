/*===------- clang/Config/clang-config.h - clang configuration ----*- C -*-===*/
/*                                                                            */
/* Part of the LLVM Project, under the Apache License v2.0 with LLVM          */
/* Exceptions.                                                                */
/* See https://llvm.org/LICENSE.txt for license information.                  */
/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    */
/*                                                                            */
/*===----------------------------------------------------------------------===*/

/* This file enumerates variables from the clang configuration so that they
   can be used in exported headers. Unlike clang/Config/config.h, it is public
   and may be included from any header. */

#ifndef CLANG_CONFIG_CLANG_CONFIG_H
#define CLANG_CONFIG_CLANG_CONFIG_H

/* Define if the export annotations (CLANG_ABI, ...) of the clang public
   interface are enabled. */
#cmakedefine CLANG_ENABLE_CLANG_EXPORT_ANNOTATIONS

#endif
