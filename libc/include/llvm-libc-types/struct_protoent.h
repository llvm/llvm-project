//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Definition of struct protoent.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TYPES_STRUCT_PROTOENT_H
#define LLVM_LIBC_TYPES_STRUCT_PROTOENT_H

struct protoent {
  char *p_name;
  char **p_aliases;
  int p_proto;
};

#endif // LLVM_LIBC_TYPES_STRUCT_PROTOENT_H
