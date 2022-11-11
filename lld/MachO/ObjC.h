//===- ObjC.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_OBJC_H
#define LLD_MACHO_OBJC_H

#include "llvm/Support/MemoryBuffer.h"

namespace lld::macho {

namespace objc {

constexpr const char klass[] = "_OBJC_CLASS_$_";
constexpr const char metaclass[] = "_OBJC_METACLASS_$_";
constexpr const char ehtype[] = "_OBJC_EHTYPE_$_";
constexpr const char ivar[] = "_OBJC_IVAR_$_";

} // namespace objc

bool hasObjCSection(llvm::MemoryBufferRef);

} // namespace lld::macho

#endif
