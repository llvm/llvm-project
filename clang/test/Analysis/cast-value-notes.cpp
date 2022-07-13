// RUN: %clang_analyze_cc1 -std=c++14 -triple amdgcn-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN: -analyzer-output=text -verify -DX86 -DSUPPRESSED %s 2>&1 | FileCheck %s -check-prefix=X86-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple amdgcn-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-config core.NullDereference:SuppressAddressSpaces=false\
// RUN:  -analyzer-output=text -verify -DX86 -DNOT_SUPPRESSED %s 2>&1 | FileCheck %s -check-prefix=X86-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple amdgcn-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-config core.NullDereference:SuppressAddressSpaces=true\
// RUN:  -analyzer-output=text -verify -DX86 -DSUPPRESSED %s 2>&1 | FileCheck %s -check-prefix=X86-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-unknown-unknown \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-output=text -verify -DX86 -DSUPPRESSED %s 2>&1 | FileCheck %s --check-prefix=X86-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-unknown-unknown \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-config core.NullDereference:SuppressAddressSpaces=true\
// RUN:  -analyzer-output=text -verify -DX86 -DSUPPRESSED %s 2>&1 | FileCheck %s --check-prefix=X86-CHECK-SUPPRESSED
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-unknown-unknown \
// RUN:  -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN:  -analyzer-config core.NullDereference:SuppressAddressSpaces=false\
// RUN:  -analyzer-output=text -verify -DX86 -DNOT_SUPPRESSED %s 2>&1 | FileCheck %s --check-prefix=X86-CHECK
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple mips-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN: -analyzer-output=text -verify -DMIPS %s 2>&1
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple mips-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN: -analyzer-config core.NullDereference:SuppressAddressSpaces=false\
// RUN: -analyzer-output=text -verify -DMIPS %s 2>&1
//
// RUN: %clang_analyze_cc1 -std=c++14 -triple mips-unknown-unknown \
// RUN: -analyzer-checker=core,apiModeling.llvm.CastValue,debug.ExprInspection\
// RUN: -analyzer-config core.NullDereference:SuppressAddressSpaces=true\
// RUN: -analyzer-output=text -verify -DMIPS_SUPPRESSED %s

#include "Inputs/llvm.h"

// The amggcn triple case uses an intentionally different address space.
// The core.NullDereference checker intentionally ignores checks
// that use address spaces, so the case is differentiated here.
//
// From https://llvm.org/docs/AMDGPUUsage.html#address-spaces,
// select address space 3 (local), since the pointer size is
// different than Generic.
#define DEVICE __attribute__((address_space(3)))

namespace clang {
struct Shape {
  template <typename T>
  const T *castAs() const;

  template <typename T>
  const T *getAs() const;
};
class Triangle : public Shape {};
class Rectangle : public Shape {};
class Hexagon : public Shape {};
class Circle : public Shape {};
} // namespace clang

using namespace llvm;
using namespace clang;

void clang_analyzer_printState();

#if defined(X86)
void evalReferences(const Shape &S) {
  const auto &C = dyn_cast<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'const class clang::Circle &'}}
  // expected-note@-2 {{Dereference of null pointer}}
  // expected-warning@-3 {{Dereference of null pointer}}
  clang_analyzer_printState();
  // XX86-CHECK:      "dynamic_types": [
  // XX86-CHECK-NEXT:   { "region": "SymRegion{reg_$0<const struct clang::Shape & S>}", "dyn_type": "const class clang::Circle &", "sub_classable": true }
  (void)C;
}
#if defined(SUPPRESSED)
void evalReferences_addrspace(const Shape &S) {
  const auto &C = dyn_cast<DEVICE Circle>(S);
  clang_analyzer_printState();
  // X86-CHECK-SUPPRESSED: "dynamic_types": [
  // X86-CHECK-SUPPRESSED-NEXT: { "region": "SymRegion{reg_$0<const struct clang::Shape & S>}", "dyn_type": "const __attribute__((address_space(3))) class clang::Circle &", "sub_classable": true }
  (void)C;
}
#endif
#if defined(NOT_SUPPRESSED)
void evalReferences_addrspace(const Shape &S) {
  const auto &C = dyn_cast<DEVICE Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'const __attribute__((address_space(3))) class clang::Circle &'}}
  // expected-note@-2 {{Dereference of null pointer}}
  // expected-warning@-3 {{Dereference of null pointer}}
  clang_analyzer_printState();
  // X86-CHECK: "dynamic_types": [
  // X86-CHECK-NEXT: { "region": "SymRegion{reg_$0<const struct clang::Shape & S>}", "dyn_type": "const __attribute__((address_space(3))) class clang::Circle &", "sub_classable": true }
  (void)C;
}
#endif
#elif defined(MIPS)
void evalReferences(const Shape &S) {
  const auto &C = dyn_cast<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'const class clang::Circle &'}}
  // expected-note@-2 {{Dereference of null pointer}}
  // expected-warning@-3 {{Dereference of null pointer}}
}
#if defined(MIPS_SUPPRESSED)
void evalReferences_addrspace(const Shape &S) {
  const auto &C = dyn_cast<DEVICE Circle>(S);
  (void)C;
}
#endif
#endif

void evalNonNullParamNonNullReturnReference(const Shape &S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{'C' initialized here}}

  if (!dyn_cast_or_null<Circle>(C)) {
    // expected-note@-1 {{Assuming 'C' is a 'const class clang::Circle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Triangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Triangle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Rectangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Rectangle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Hexagon>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Hexagon *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle' nor a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Circle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is a 'Circle'}}
    // expected-note@-2 {{Taking true branch}}

    (void)(1 / !C);
    // expected-note@-1 {{'C' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
}

void evalNonNullParamNonNullReturn(const Shape *S) {
  const auto *C = cast<Circle>(S);
  // expected-note@-1 {{'S' is a 'const class clang::Circle *'}}
  // expected-note@-2 {{'C' initialized here}}

  if (!dyn_cast_or_null<Circle>(C)) {
    // expected-note@-1 {{Assuming 'C' is a 'const class clang::Circle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Triangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Triangle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Rectangle>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Rectangle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (dyn_cast_or_null<Hexagon>(C)) {
    // expected-note@-1 {{Assuming 'C' is not a 'const class clang::Hexagon *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle>(C)) {
    // expected-note@-1 {{'C' is not a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Triangle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is neither a 'Triangle' nor a 'Rectangle' nor a 'Hexagon'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (isa<Circle, Rectangle, Hexagon>(C)) {
    // expected-note@-1 {{'C' is a 'Circle'}}
    // expected-note@-2 {{Taking true branch}}

    (void)(1 / !C);
    // expected-note@-1 {{'C' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
}

void evalNonNullParamNullReturn(const Shape *S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{Assuming 'S' is not a 'const class clang::Circle *'}}

  if (const auto *T = dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{Assuming 'S' is a 'const class clang::Triangle *'}}
    // expected-note@-2 {{'T' initialized here}}
    // expected-note@-3 {{'T' is non-null}}
    // expected-note@-4 {{Taking true branch}}

    (void)(1 / !T);
    // expected-note@-1 {{'T' is non-null}}
    // expected-note@-2 {{Division by zero}}
    // expected-warning@-3 {{Division by zero}}
  }
}

void evalNullParamNullReturn(const Shape *S) {
  const auto *C = dyn_cast_or_null<Circle>(S);
  // expected-note@-1 {{Assuming null pointer is passed into cast}}
  // expected-note@-2 {{'C' initialized to a null pointer value}}

  (void)(1 / (bool)C);
  // expected-note@-1 {{Division by zero}}
  // expected-warning@-2 {{Division by zero}}
}

void evalZeroParamNonNullReturnPointer(const Shape *S) {
  const auto *C = S->castAs<Circle>();
  // expected-note@-1 {{'S' is a 'const class clang::Circle *'}}
  // expected-note@-2 {{'C' initialized here}}

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

void evalZeroParamNonNullReturn(const Shape &S) {
  const auto *C = S.castAs<Circle>();
  // expected-note@-1 {{'C' initialized here}}

  (void)(1 / !C);
  // expected-note@-1 {{'C' is non-null}}
  // expected-note@-2 {{Division by zero}}
  // expected-warning@-3 {{Division by zero}}
}

void evalZeroParamNullReturn(const Shape *S) {
  const auto &C = S->getAs<Circle>();
  // expected-note@-1 {{Assuming 'S' is not a 'const class clang::Circle *'}}
  // expected-note@-2 {{Storing null pointer value}}
  // expected-note@-3 {{'C' initialized here}}

  if (!dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{Assuming 'S' is a 'const class clang::Triangle *'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  if (!dyn_cast_or_null<Triangle>(S)) {
    // expected-note@-1 {{'S' is a 'Triangle'}}
    // expected-note@-2 {{Taking false branch}}
    return;
  }

  (void)(1 / (bool)C);
  // expected-note@-1 {{Division by zero}}
  // expected-warning@-2 {{Division by zero}}
}

// don't crash
// CastValueChecker was using QualType()->getPointeeCXXRecordDecl(), in
// getNoteTag which evaluated to nullptr, then crashed when attempting to
// deref an invocation to getNameAsString(). The fix is to use
// QualType().getAsString().
//
// Example:
// std::string CastToName =
//       CastInfo ? CastInfo->to()->getAsCXXRecordDecl()->getNameAsString()
//                : CastToTy->getPointeeCXXRecordDecl()->getNameAsString();
// Changed to:
// std::string CastToName =
//       CastInfo ? CastInfo->to()->getAsCXXRecordDecl()->getNameAsString()
//                : CastToTy.getAsString();
namespace llvm {
template <typename, typename a> void isa(a &);
template <typename> class PointerUnion {
public:
  template <typename T> T *getAs() {
    (void)isa<int>(*this);
    return nullptr;
  }
};
class LLVMContext {
  PointerUnion<LLVMContext> c;
  void d() { c.getAs<int>(); }
};
} // namespace llvm
