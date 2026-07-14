//===--- ExprConstShared.h - Shared consetxpr functionality ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared functionality between the new constant expression
// interpreter (AST/ByteCode/) and the current one (ExprConstant.cpp).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_AST_EXPRCONSTSHARED_H
#define LLVM_CLANG_LIB_AST_EXPRCONSTSHARED_H

#include "clang/AST/CharUnits.h"
#include "clang/Basic/TypeTraits.h"
#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace llvm {
class APFloat;
class APSInt;
class APInt;
}
namespace clang {
class QualType;
class LangOptions;
class ASTContext;
class APValue;
class ConstantArrayType;
class Expr;
class CallExpr;
class StringLiteral;
} // namespace clang
using namespace clang;
/// Values returned by __builtin_classify_type, chosen to match the values
/// produced by GCC's builtin.
enum class GCCTypeClass {
  None = -1,
  Void = 0,
  Integer = 1,
  // GCC reserves 2 for character types, but instead classifies them as
  // integers.
  Enum = 3,
  Bool = 4,
  Pointer = 5,
  // GCC reserves 6 for references, but appears to never use it (because
  // expressions never have reference type, presumably).
  PointerToDataMember = 7,
  RealFloat = 8,
  Complex = 9,
  // GCC reserves 10 for functions, but does not use it since GCC version 6 due
  // to decay to pointer. (Prior to version 6 it was only used in C++ mode).
  // GCC claims to reserve 11 for pointers to member functions, but *actually*
  // uses 12 for that purpose, same as for a class or struct. Maybe it
  // internally implements a pointer to member as a struct?  Who knows.
  PointerToMemberFunction = 12, // Not a bug, see above.
  ClassOrStruct = 12,
  Union = 13,
  // GCC reserves 14 for arrays, but does not use it since GCC version 6 due to
  // decay to pointer. (Prior to version 6 it was only used in C++ mode).
  // GCC reserves 15 for strings, but actually uses 5 (pointer) for string
  // literals.
  // Lang = 16,
  // OpaqueType = 17,
  BitInt = 18,
  Vector = 19
};

GCCTypeClass EvaluateBuiltinClassifyType(QualType T,
                                         const LangOptions &LangOpts);

void HandleComplexComplexMul(llvm::APFloat A, llvm::APFloat B, llvm::APFloat C,
                             llvm::APFloat D, llvm::APFloat &ResR,
                             llvm::APFloat &ResI);
void HandleComplexComplexDiv(llvm::APFloat A, llvm::APFloat B, llvm::APFloat C,
                             llvm::APFloat D, llvm::APFloat &ResR,
                             llvm::APFloat &ResI);

CharUnits GetAlignOfExpr(const ASTContext &Ctx, const Expr *E,
                         UnaryExprOrTypeTrait ExprKind);

/// Convert a builtin ID to the canonical x86 builtin ID the constant evaluators
/// dispatch on in their x86 target-specific cases.
///
/// Target-independent builtins are returned unchanged. An x86 target builtin
/// (including an auxiliary-target x86 builtin, whose ID is shifted past the
/// primary target's builtins) is translated to its canonical X86::BI* value.
/// Any other target's builtin returns 0: the constant evaluators only fold x86
/// target builtins, and target builtin IDs of different targets overlap (each
/// numbers from Builtin::FirstTSBuiltin), so an unrelated target's ID must not
/// be mistaken for an x86 one.
///
/// The ID-based overload performs no work beyond a single comparison for
/// target-independent builtins, so it is suitable for hot paths (e.g. the
/// bytecode interpreter's builtin dispatch) where re-deriving the ID from the
/// call expression would be wasteful.
unsigned ConvertBuiltinIDToX86BuiltinID(const ASTContext &Ctx,
                                        unsigned BuiltinID);
unsigned ConvertBuiltinIDToX86BuiltinID(const ASTContext &Ctx,
                                        const CallExpr *E);
uint8_t GFNIMultiplicativeInverse(uint8_t Byte);
uint8_t GFNIMul(uint8_t AByte, uint8_t BByte);
uint8_t GFNIAffine(uint8_t XByte, const llvm::APInt &AQword,
                   const llvm::APSInt &Imm, bool Inverse = false);
llvm::APSInt NormalizeRotateAmount(const llvm::APSInt &Value,
                                   const llvm::APSInt &Amount);

std::optional<llvm::APFloat>
EvalScalarMinMaxFp(const llvm::APFloat &A, const llvm::APFloat &B,
                   std::optional<llvm::APSInt> RoundingMode, bool IsMin);

/// Where an lvalue into an array element lives: the element index within the
/// array (or the array length for a one-past-the-end pointer), and the byte
/// offset from the start of that element.
struct ArraySubobjectLocation {
  uint64_t Index;
  CharUnits OffsetInElement;
};

/// Computes the array-element location designated by an lvalue whose first
/// path entry indexes into ArrayType with the given Index and whose
/// byte offset from the array base is LValueOffset. IsValidOnePastEnd
/// must be true iff the lvalue is a valid one-past-the-end position of the
/// array (which the caller determines from its own lvalue representation).
/// Returns std::nullopt if the lvalue does not designate an element,
/// one-past-the-end position, or subobject of an element.
std::optional<ArraySubobjectLocation> computeArraySubobjectLocation(
    const ASTContext &Ctx, const ConstantArrayType *ArrayType, uint64_t Index,
    CharUnits LValueOffset, bool IsValidOnePastEnd);

/// A potentially-non-unique array object: a string literal, an
/// std::initializer_list backing array, or an array subobject of a
/// template-parameter object (NTTP). [intro.object]/9 lets the implementation
/// merge two such objects when their overlapping contents agree, which is what
/// makes their address comparisons non-constant.
///
/// This abstract class is an evaluator-independent view of an object that
/// [intro.object] permits an implementation to give the same address as another
/// potentially non-unique object. Evaluators provide their own object
/// recognition, subobject-location logic, and element materialization through
/// derived classes.
class PotentiallyNonUniqueObject {
public:
  enum class Kind { StringLiteral, InitializerList, TemplateParamObject };
  enum class OverlapResult { None, StringLiteral, NonUniqueObject };

  virtual ~PotentiallyNonUniqueObject() = default;

  Kind kind() const {
    assert(ArrayType && "object has not been recognized");
    return TheKind;
  }
  bool isStringLiteral() const { return String.has_value(); }
  const ConstantArrayType *arrayType() const { return ArrayType; }
  const ArraySubobjectLocation &location() const { return Loc; }
  uint64_t size() const { return Size; }
  bool empty() const { return Size == 0; }

  /// Determine whether two expressions denote string-like objects whose
  /// storage can overlap at the given byte offsets. Returns std::nullopt if
  /// either expression is not a string literal, predefined function-name
  /// string, or Objective-C encoding.
  static std::optional<bool>
  isPotentiallyOverlappingStrings(const ASTContext &Ctx, const Expr *LHSBase,
                                  CharUnits LHSOffset, const Expr *RHSBase,
                                  CharUnits RHSOffset);

  /// Returns true if this object and \p Other can share storage at their
  /// designated positions. Missing element data is treated conservatively as
  /// a possible match.
  bool
  isPotentiallyOverlappingWith(const PotentiallyNonUniqueObject &Other) const;

protected:
  explicit PotentiallyNonUniqueObject(const ASTContext &Ctx) : Ctx(Ctx) {}

  /// Computes an array-element location from only a byte offset relative to
  /// the array. This is used by evaluator adapters whose lvalue representation
  /// does not carry an explicit path for the designated string subobject.
  static std::optional<ArraySubobjectLocation>
  computeArraySubobjectLocationFromOffset(const ASTContext &Ctx,
                                          const ConstantArrayType *ArrayType,
                                          CharUnits LValueOffset);

  PotentiallyNonUniqueObject(const ASTContext &Ctx, Kind TheKind,
                             const ConstantArrayType *ArrayType,
                             ArraySubobjectLocation Loc, uint64_t Size)
      : Ctx(Ctx) {
    setRecognizedObject(TheKind, ArrayType, Loc, Size);
  }

  void setRecognizedObject(Kind NewKind, const ConstantArrayType *NewArrayType,
                           ArraySubobjectLocation NewLoc, uint64_t NewSize) {
    assert(NewArrayType && !ArrayType && "object must be recognized once");
    assert((NewKind == Kind::StringLiteral) == String.has_value() &&
           "string kind and storage disagree");
    TheKind = NewKind;
    ArrayType = NewArrayType;
    Loc = NewLoc;
    Size = NewSize;
  }

  /// Recognize Base as a StringLiteral or PredefinedExpr and install its
  /// borrowed AST storage together with the evaluator-computed array view.
  /// Objective-C encodings are intentionally limited to
  /// isPotentiallyOverlappingStrings; they are not exposed as generic array
  /// objects here.
  bool setRecognizedStringObject(const Expr *Base,
                                 const ConstantArrayType *NewArrayType,
                                 ArraySubobjectLocation NewLoc,
                                 uint64_t NewSize, CharUnits RawOffset);

  /// Materialize an array element from the evaluator's storage representation.
  /// Returns false if the representation cannot expose enough data to prove
  /// the element's value.
  virtual bool getArrayElement(uint64_t Index, APValue &Result) const = 0;

  const ASTContext &Ctx;

private:
  /// Owns string-like bytes synthesized during normalization (currently for
  /// ObjCEncodeExpr), together with their code-unit width in bytes.
  struct OwnedString {
    std::string Bytes;
    unsigned CharWidth;
  };

  /// Normalized storage and location for a string-like object. StringLiteral
  /// storage is borrowed for the ASTContext lifetime, while synthesized bytes
  /// are owned. Offset is the designated pointer's byte offset from the string
  /// base; it remains separate from the evaluator-specific array location so
  /// strings with different code-unit widths can be compared.
  struct StringData {
    StringData(const StringLiteral *Literal, CharUnits Offset)
        : Storage(Literal), Offset(Offset) {}
    StringData(std::string Bytes, unsigned CharWidth, CharUnits Offset)
        : Storage(OwnedString{std::move(Bytes), CharWidth}), Offset(Offset) {}

    std::variant<const StringLiteral *, OwnedString> Storage;
    CharUnits Offset;
  };

  static std::optional<StringData>
  getStringData(const ASTContext &Ctx, const Expr *Base, CharUnits Offset);
  static bool isPotentiallyOverlappingStrings(const StringData &LHS,
                                              const StringData &RHS);

  /// Whether two APValues could be merged into a single storage location by
  /// the implementation. The slow path recursively handles values whose
  /// profiles differ.
  static bool areAPValuesPotentiallyMergeable(const APValue &LHS,
                                              const APValue &RHS,
                                              const ASTContext &Ctx);
  static bool areAPValuesPotentiallyMergeableSlow(const APValue &LHS,
                                                  const APValue &RHS,
                                                  const ASTContext &Ctx);

  /// Materialize element Index, handling strings in the shared base before
  /// delegating evaluator-backed objects to getArrayElement.
  bool getElement(uint64_t Index, APValue &Result) const;

  Kind TheKind = Kind::StringLiteral;
  const ConstantArrayType *ArrayType = nullptr;
  ArraySubobjectLocation Loc{0, CharUnits::Zero()};
  uint64_t Size = 0;
  std::optional<StringData> String;
};

#endif // LLVM_CLANG_LIB_AST_EXPRCONSTSHARED_H
