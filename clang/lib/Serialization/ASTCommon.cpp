//===--- ASTCommon.cpp - Common stuff for ASTReader/ASTWriter----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#include "ASTCommon.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;

// Give ASTDeserializationListener's VTable a home.
ASTDeserializationListener::~ASTDeserializationListener() { }

serialization::TypeIdx
serialization::TypeIdxFromBuiltin(const BuiltinType *BT) {
  unsigned ID = 0;
  switch (BT->getKind()) {
  case BuiltinType::Void:       ID = PREDEF_TYPE_VOID_ID;       break;
  case BuiltinType::Bool:       ID = PREDEF_TYPE_BOOL_ID;       break;
  case BuiltinType::Char_U:     ID = PREDEF_TYPE_CHAR_U_ID;     break;
  case BuiltinType::UChar:      ID = PREDEF_TYPE_UCHAR_ID;      break;
  case BuiltinType::UShort:     ID = PREDEF_TYPE_USHORT_ID;     break;
  case BuiltinType::UInt:       ID = PREDEF_TYPE_UINT_ID;       break;
  case BuiltinType::ULong:      ID = PREDEF_TYPE_ULONG_ID;      break;
  case BuiltinType::ULongLong:  ID = PREDEF_TYPE_ULONGLONG_ID;  break;
  case BuiltinType::UInt128:    ID = PREDEF_TYPE_UINT128_ID;    break;
  case BuiltinType::Char_S:     ID = PREDEF_TYPE_CHAR_S_ID;     break;
  case BuiltinType::SChar:      ID = PREDEF_TYPE_SCHAR_ID;      break;
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:    ID = PREDEF_TYPE_WCHAR_ID;      break;
  case BuiltinType::Short:      ID = PREDEF_TYPE_SHORT_ID;      break;
  case BuiltinType::Int:        ID = PREDEF_TYPE_INT_ID;        break;
  case BuiltinType::Long:       ID = PREDEF_TYPE_LONG_ID;       break;
  case BuiltinType::LongLong:   ID = PREDEF_TYPE_LONGLONG_ID;   break;
  case BuiltinType::Int128:     ID = PREDEF_TYPE_INT128_ID;     break;
  case BuiltinType::Half:       ID = PREDEF_TYPE_HALF_ID;       break;
  case BuiltinType::Float:      ID = PREDEF_TYPE_FLOAT_ID;      break;
  case BuiltinType::Double:     ID = PREDEF_TYPE_DOUBLE_ID;     break;
  case BuiltinType::LongDouble: ID = PREDEF_TYPE_LONGDOUBLE_ID; break;
  case BuiltinType::NullPtr:    ID = PREDEF_TYPE_NULLPTR_ID;    break;
  case BuiltinType::Char16:     ID = PREDEF_TYPE_CHAR16_ID;     break;
  case BuiltinType::Char32:     ID = PREDEF_TYPE_CHAR32_ID;     break;
  case BuiltinType::Overload:   ID = PREDEF_TYPE_OVERLOAD_ID;   break;
  case BuiltinType::BoundMember:ID = PREDEF_TYPE_BOUND_MEMBER;  break;
  case BuiltinType::PseudoObject:ID = PREDEF_TYPE_PSEUDO_OBJECT;break;
  case BuiltinType::Dependent:  ID = PREDEF_TYPE_DEPENDENT_ID;  break;
  case BuiltinType::UnknownAny: ID = PREDEF_TYPE_UNKNOWN_ANY;   break;
  case BuiltinType::ARCUnbridgedCast:
                                ID = PREDEF_TYPE_ARC_UNBRIDGED_CAST; break;
  case BuiltinType::ObjCId:     ID = PREDEF_TYPE_OBJC_ID;       break;
  case BuiltinType::ObjCClass:  ID = PREDEF_TYPE_OBJC_CLASS;    break;
  case BuiltinType::ObjCSel:    ID = PREDEF_TYPE_OBJC_SEL;      break;
  case BuiltinType::BuiltinFn:
                                ID = PREDEF_TYPE_BUILTIN_FN; break;

  }

  return TypeIdx(ID);
}

unsigned serialization::ComputeHash(Selector Sel) {
  unsigned N = Sel.getNumArgs();
  if (N == 0)
    ++N;
  unsigned R = 5381;
  for (unsigned I = 0; I != N; ++I)
    if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
      R = llvm::HashString(II->getName(), R);
  return R;
}
