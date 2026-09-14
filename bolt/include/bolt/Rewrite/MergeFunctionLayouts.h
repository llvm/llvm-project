//===- bolt/Rewrite/MergeFunctionLayouts.h ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_REWRITE_MERGEFUNCTIONLAYOUTS_H
#define BOLT_REWRITE_MERGEFUNCTIONLAYOUTS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;

namespace bolt {

/// Merge two layout files \p PathA and \p PathB into an aligned layout stored
/// to \p OutputPath. Only common functions that occur in the same relative
/// order are included.
Error mergeFunctionLayouts(StringRef PathA, StringRef PathB,
                           StringRef OutputPath, raw_ostream &Log);

} // namespace bolt
} // namespace llvm

#endif
