//===- TGRegion.h - TableGen region definitions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TABLEGEN_REGION_H_
#define AIIR_TABLEGEN_REGION_H_

#include "aiir/Support/LLVM.h"
#include "aiir/TableGen/Constraint.h"

namespace aiir {
namespace tblgen {

// Wrapper class providing helper methods for accessing Region defined in
// TableGen.
class Region : public Constraint {
public:
  using Constraint::Constraint;

  static bool classof(const Constraint *c) { return c->getKind() == CK_Region; }

  // Returns true if this region is variadic.
  bool isVariadic() const;
};

// A struct bundling a region's constraint and its name.
struct NamedRegion {
  // Returns true if this region is variadic.
  bool isVariadic() const { return constraint.isVariadic(); }

  StringRef name;
  Region constraint;
};

} // namespace tblgen
} // namespace aiir

#endif // AIIR_TABLEGEN_REGION_H_
