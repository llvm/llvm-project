//=== ContainerOfModeling.h ----------------------------------------*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_CONTAINEROFMODELING_H
#define LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_CONTAINEROFMODELING_H

#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"

namespace clang::ento {

class ElementRegion;
class SubRegion;
class SValBuilder;

/// Recognize the region shape produced when a pointer to a direct field is
/// adjusted back to the beginning of its containing record. For example,
///
///   (struct Parent *)((char *)&P.Field - offsetof(struct Parent, Field))
///
/// is represented as:
///
///   ElementRegion<Parent, 0>
///     ElementRegion<char, -offsetof(Parent, Field)>
///       FieldRegion<Parent::Field>
///         <region for P>
///
/// The character ElementRegion is absent when the field offset is zero. Return
/// the region for P only when the record type, field declaration, target ABI
/// layout, and underlying storage prove that the adjustment lands exactly at
/// the beginning of P.
const SubRegion *getContainerOfParentRegion(const ElementRegion *ContainerER,
                                            ProgramStateRef State,
                                            SValBuilder &SVB);

} // namespace clang::ento

#endif // LLVM_CLANG_LIB_STATICANALYZER_CHECKERS_CONTAINEROFMODELING_H
