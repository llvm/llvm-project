//===--- NSAPI.h - NSFoundation APIs ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_NSAPI_H
#define LLVM_CLANG_AST_NSAPI_H

#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace clang {
  class ASTContext;
  class QualType;
  class Expr;

// \brief Provides info and caches identifiers/selectors for NSFoundation API.
class NSAPI {
public:
  explicit NSAPI(ASTContext &Ctx);

  ASTContext &getASTContext() const { return Ctx; }

  enum NSClassIdKindKind {
    ClassId_NSObject,
    ClassId_NSString,
    ClassId_NSArray,
    ClassId_NSMutableArray,
    ClassId_NSDictionary,
    ClassId_NSMutableDictionary,
    ClassId_NSNumber
  };
  static const unsigned NumClassIds = 7;

  enum NSStringMethodKind {
    NSStr_stringWithString,
    NSStr_stringWithUTF8String,
    NSStr_stringWithCStringEncoding,
    NSStr_stringWithCString,
    NSStr_initWithString
  };
  static const unsigned NumNSStringMethods = 5;

  IdentifierInfo *getNSClassId(NSClassIdKindKind K) const;

  /// \brief The Objective-C NSString selectors.
  Selector getNSStringSelector(NSStringMethodKind MK) const;

  /// \brief Return NSStringMethodKind if \param Sel is such a selector.
  llvm::Optional<NSStringMethodKind> getNSStringMethodKind(Selector Sel) const;

  /// \brief Returns true if the expression \param E is a reference of
  /// "NSUTF8StringEncoding" enum constant.
  bool isNSUTF8StringEncodingConstant(const Expr *E) const {
    return isObjCEnumerator(E, "NSUTF8StringEncoding", NSUTF8StringEncodingId);
  }

  /// \brief Returns true if the expression \param E is a reference of
  /// "NSASCIIStringEncoding" enum constant.
  bool isNSASCIIStringEncodingConstant(const Expr *E) const {
    return isObjCEnumerator(E, "NSASCIIStringEncoding",NSASCIIStringEncodingId);
  }

  /// \brief Enumerates the NSArray methods used to generate literals.
  enum NSArrayMethodKind {
    NSArr_array,
    NSArr_arrayWithArray,
    NSArr_arrayWithObject,
    NSArr_arrayWithObjects,
    NSArr_arrayWithObjectsCount,
    NSArr_initWithArray,
    NSArr_initWithObjects,
    NSArr_objectAtIndex,
    NSMutableArr_replaceObjectAtIndex
  };
  static const unsigned NumNSArrayMethods = 9;

  /// \brief The Objective-C NSArray selectors.
  Selector getNSArraySelector(NSArrayMethodKind MK) const;

  /// \brief Return NSArrayMethodKind if \p Sel is such a selector.
  llvm::Optional<NSArrayMethodKind> getNSArrayMethodKind(Selector Sel);

  /// \brief Enumerates the NSDictionary methods used to generate literals.
  enum NSDictionaryMethodKind {
    NSDict_dictionary,
    NSDict_dictionaryWithDictionary,
    NSDict_dictionaryWithObjectForKey,
    NSDict_dictionaryWithObjectsForKeys,
    NSDict_dictionaryWithObjectsForKeysCount,
    NSDict_dictionaryWithObjectsAndKeys,
    NSDict_initWithDictionary,
    NSDict_initWithObjectsAndKeys,
    NSDict_objectForKey,
    NSMutableDict_setObjectForKey
  };
  static const unsigned NumNSDictionaryMethods = 10;

  /// \brief The Objective-C NSDictionary selectors.
  Selector getNSDictionarySelector(NSDictionaryMethodKind MK) const;

  /// \brief Return NSDictionaryMethodKind if \p Sel is such a selector.
  llvm::Optional<NSDictionaryMethodKind>
      getNSDictionaryMethodKind(Selector Sel);

  /// \brief Returns selector for "objectForKeyedSubscript:".
  Selector getObjectForKeyedSubscriptSelector() const {
    return getOrInitSelector(StringRef("objectForKeyedSubscript"),
                             objectForKeyedSubscriptSel);
  }

  /// \brief Returns selector for "objectAtIndexedSubscript:".
  Selector getObjectAtIndexedSubscriptSelector() const {
    return getOrInitSelector(StringRef("objectAtIndexedSubscript"),
                             objectAtIndexedSubscriptSel);
  }

  /// \brief Returns selector for "setObject:forKeyedSubscript".
  Selector getSetObjectForKeyedSubscriptSelector() const {
    StringRef Ids[] = { "setObject", "forKeyedSubscript" };
    return getOrInitSelector(Ids, setObjectForKeyedSubscriptSel);
  }

  /// \brief Returns selector for "setObject:atIndexedSubscript".
  Selector getSetObjectAtIndexedSubscriptSelector() const {
    StringRef Ids[] = { "setObject", "atIndexedSubscript" };
    return getOrInitSelector(Ids, setObjectAtIndexedSubscriptSel);
  }

  /// \brief Returns selector for "isEqual:".
  Selector getIsEqualSelector() const {
    return getOrInitSelector(StringRef("isEqual"), isEqualSel);
  }

  /// \brief Enumerates the NSNumber methods used to generate literals.
  enum NSNumberLiteralMethodKind {
    NSNumberWithChar,
    NSNumberWithUnsignedChar,
    NSNumberWithShort,
    NSNumberWithUnsignedShort,
    NSNumberWithInt,
    NSNumberWithUnsignedInt,
    NSNumberWithLong,
    NSNumberWithUnsignedLong,
    NSNumberWithLongLong,
    NSNumberWithUnsignedLongLong,
    NSNumberWithFloat,
    NSNumberWithDouble,
    NSNumberWithBool,
    NSNumberWithInteger,
    NSNumberWithUnsignedInteger
  };
  static const unsigned NumNSNumberLiteralMethods = 15;

  /// \brief The Objective-C NSNumber selectors used to create NSNumber literals.
  /// \param Instance if true it will return the selector for the init* method
  /// otherwise it will return the selector for the number* method.
  Selector getNSNumberLiteralSelector(NSNumberLiteralMethodKind MK,
                                      bool Instance) const;

  bool isNSNumberLiteralSelector(NSNumberLiteralMethodKind MK,
                                 Selector Sel) const {
    return Sel == getNSNumberLiteralSelector(MK, false) ||
           Sel == getNSNumberLiteralSelector(MK, true);
  }

  /// \brief Return NSNumberLiteralMethodKind if \p Sel is such a selector.
  llvm::Optional<NSNumberLiteralMethodKind>
      getNSNumberLiteralMethodKind(Selector Sel) const;

  /// \brief Determine the appropriate NSNumber factory method kind for a
  /// literal of the given type.
  llvm::Optional<NSNumberLiteralMethodKind>
      getNSNumberFactoryMethodKind(QualType T) const;

  /// \brief Returns true if \param T is a typedef of "BOOL" in objective-c.
  bool isObjCBOOLType(QualType T) const;
  /// \brief Returns true if \param T is a typedef of "NSInteger" in objective-c.
  bool isObjCNSIntegerType(QualType T) const;
  /// \brief Returns true if \param T is a typedef of "NSUInteger" in objective-c.
  bool isObjCNSUIntegerType(QualType T) const;

private:
  bool isObjCTypedef(QualType T, StringRef name, IdentifierInfo *&II) const;
  bool isObjCEnumerator(const Expr *E,
                        StringRef name, IdentifierInfo *&II) const;
  Selector getOrInitSelector(ArrayRef<StringRef> Ids, Selector &Sel) const;

  ASTContext &Ctx;

  mutable IdentifierInfo *ClassIds[NumClassIds];

  mutable Selector NSStringSelectors[NumNSStringMethods];

  /// \brief The selectors for Objective-C NSArray methods.
  mutable Selector NSArraySelectors[NumNSArrayMethods];

  /// \brief The selectors for Objective-C NSDictionary methods.
  mutable Selector NSDictionarySelectors[NumNSDictionaryMethods];

  /// \brief The Objective-C NSNumber selectors used to create NSNumber literals.
  mutable Selector NSNumberClassSelectors[NumNSNumberLiteralMethods];
  mutable Selector NSNumberInstanceSelectors[NumNSNumberLiteralMethods];

  mutable Selector objectForKeyedSubscriptSel, objectAtIndexedSubscriptSel,
                   setObjectForKeyedSubscriptSel,setObjectAtIndexedSubscriptSel,
                   isEqualSel;

  mutable IdentifierInfo *BOOLId, *NSIntegerId, *NSUIntegerId;
  mutable IdentifierInfo *NSASCIIStringEncodingId, *NSUTF8StringEncodingId;
};

}  // end namespace clang

#endif // LLVM_CLANG_AST_NSAPI_H
