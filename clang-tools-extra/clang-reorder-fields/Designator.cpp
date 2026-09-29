//===-- tools/extra/clang-reorder-fields/utils/Designator.cpp ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the Designator and Designators utility
/// classes.
///
//===----------------------------------------------------------------------===//

#include "Designator.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace reorder_fields {

void Designator::advanceToNextField() {
  assert(!isFinished() && "Iterator is already finished");
  switch (Tag) {
  case STRUCT:
    if (StructIt.Record->isUnion()) {
      // Union always finishes on first increment.
      StructIt.Field = StructIt.Record->field_end();
      Type = QualType();
      break;
    }
    ++StructIt.Field;
    if (StructIt.Field != StructIt.Record->field_end()) {
      Type = StructIt.Field->getType();
    } else {
      Type = QualType();
    }
    break;
  case ARRAY:
    ++ArrayIt.Index;
    break;
  case ARRAY_RANGE:
    ArrayIt.Index = ArrayRangeIt.End + 1;
    ArrayIt.Size = ArrayRangeIt.Size;
    Tag = ARRAY;
    break;
  }
}

bool Designator::isFinished() {
  switch (Tag) {
  case STRUCT:
    return StructIt.Field == StructIt.Record->field_end();
  case ARRAY:
    return ArrayIt.Index == ArrayIt.Size;
  case ARRAY_RANGE:
    return ArrayRangeIt.End == ArrayRangeIt.Size;
  }
  return false;
}

Designators::Designators(const Expr *Init, const InitListExpr *ILE,
                         const ASTContext *Context)
    : Context(Context) {
  if (ILE->getType()->isArrayType()) {
    const ConstantArrayType *CAT =
        Context->getAsConstantArrayType(ILE->getType());
    // Only constant size arrays are supported.
    if (!CAT) {
      DesignatorList.clear();
      return;
    }
    DesignatorList.push_back(
        {CAT->getElementType(), 0, CAT->getSize().getZExtValue()});
  } else {
    const RecordDecl *DesignatorRD = ILE->getType()->getAsRecordDecl();
    DesignatorList.push_back({DesignatorRD->field_begin()->getType(),
                              DesignatorRD->field_begin(), DesignatorRD});
  }

  // If the designator list is empty at this point, then there must be excess
  // elements in the initializer list. They are not currently supported.
  if (DesignatorList.empty())
    return;

  if (!enterImplicitInitLists(Init))
    DesignatorList.clear();
}

Designators::Designators(const DesignatedInitExpr *DIE, const InitListExpr *ILE,
                         const ASTContext *Context)
    : Context(Context) {
  for (const auto &D : DIE->designators()) {
    if (D.isFieldDesignator()) {
      RecordDecl *DesignatorRecord = D.getFieldDecl()->getParent();
      for (auto FieldIt = DesignatorRecord->field_begin();
           FieldIt != DesignatorRecord->field_end(); ++FieldIt) {
        if (*FieldIt == D.getFieldDecl()) {
          DesignatorList.push_back(
              {FieldIt->getType(), FieldIt, DesignatorRecord});
          break;
        }
      }
    } else {
      const QualType CurrentType = DesignatorList.empty()
                                       ? ILE->getType()
                                       : DesignatorList.back().getType();
      const ConstantArrayType *CAT =
          Context->getAsConstantArrayType(CurrentType);
      if (!CAT) {
        // Non-constant-sized arrays are not supported.
        DesignatorList.clear();
        return;
      }
      if (D.isArrayDesignator()) {
        DesignatorList.push_back({CAT->getElementType(),
                                  DIE->getArrayIndex(D)
                                      ->EvaluateKnownConstInt(*Context)
                                      .getZExtValue(),
                                  CAT->getSize().getZExtValue()});
      } else if (D.isArrayRangeDesignator()) {
        DesignatorList.push_back({CAT->getElementType(),
                                  DIE->getArrayRangeStart(D)
                                      ->EvaluateKnownConstInt(*Context)
                                      .getZExtValue(),
                                  DIE->getArrayRangeEnd(D)
                                      ->EvaluateKnownConstInt(*Context)
                                      .getZExtValue(),
                                  CAT->getSize().getZExtValue()});
      } else {
        llvm_unreachable("Unexpected designator kind");
      }
    }
  }
}

bool Designators::advanceToNextField(const Expr *Init) {
  // Remove all designators that refer to the last field of a struct or final
  // element of the array.
  while (!DesignatorList.empty()) {
    auto &CurrentDesignator = DesignatorList.back();
    CurrentDesignator.advanceToNextField();
    if (CurrentDesignator.isFinished()) {
      DesignatorList.pop_back();
      continue;
    }
    break;
  }

  // If the designator list is empty at this point, then there must be excess
  // elements in the initializer list. They are not currently supported.
  if (DesignatorList.empty())
    return false;

  if (!enterImplicitInitLists(Init)) {
    DesignatorList.clear();
    return false;
  }

  return true;
}

bool Designators::enterImplicitInitLists(const Expr *Init) {
  // Check for missing braces by comparing the type of the last designator and
  // type of Init.
  while (true) {
    const QualType T = DesignatorList.back().getType();
    // If the types match, there are no missing braces.
    if (Init->getType() == T)
      break;

    // If the current type is a struct, then get its first field.
    if (T->isRecordType()) {
      DesignatorList.push_back({T->getAsRecordDecl()->field_begin()->getType(),
                                T->getAsRecordDecl()->field_begin(),
                                T->getAsRecordDecl()});
      continue;
    }
    // If the current type is an array, then get its first element.
    if (T->isArrayType()) {
      DesignatorList.push_back(
          {Context->getAsArrayType(T)->getElementType(), 0,
           Context->getAsConstantArrayType(T)->getSize().getZExtValue()});
      continue;
    }

    // The initializer doesn't match the expected type. The initializer list is
    // invalid.
    return false;
  }

  return true;
}

std::string Designators::toString() const {
  if (DesignatorList.empty())
    return "";
  std::string Designator;
  llvm::raw_string_ostream OS(Designator);
  for (auto &I : DesignatorList) {
    switch (I.getTag()) {
    case Designator::STRUCT:
      OS << '.' << I.getStructIter()->getName();
      break;
    case Designator::ARRAY:
      OS << '[' << I.getArrayIndex() << ']';
      break;
    case Designator::ARRAY_RANGE:
      OS << '[' << I.getArrayRangeStart() << "..." << I.getArrayRangeEnd()
         << ']';
    }
  }
  OS << " = ";
  return Designator;
}

} // namespace reorder_fields
} // namespace clang
