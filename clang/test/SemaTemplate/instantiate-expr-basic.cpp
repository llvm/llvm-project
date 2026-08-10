// RUN: %clang_cc1 -fsyntax-only -Wno-unused-value -std=c++11 %s

template <typename T>
struct S {
  void f() {
    __func__; // PredefinedExpr
    10;       // IntegerLiteral
    10.5;     // FloatingLiteral
    'c';      // CharacterLiteral
    "hello";  // StringLiteral
    true;     // CXXBooleanLiteralExpr
    nullptr;  // CXXNullPtrLiteralExpr
    __null;   // GNUNullExpr
  }
};

template struct S<int>;
