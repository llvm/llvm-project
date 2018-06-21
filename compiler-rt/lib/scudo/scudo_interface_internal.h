//===-- scudo_interface_internal.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Private Scudo interface header.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_INTERFACE_INTERNAL_H_
#define SCUDO_INTERFACE_INTERNAL_H_

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
void __scudo_set_rss_limit(unsigned long LimitMb, int HardLimit);  // NOLINT
}  // extern "C"

#endif  // SCUDO_INTERFACE_INTERNAL_H_
