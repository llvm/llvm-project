// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Crash from LLVM build with ClangIR. Creduced from llvm::formLCSSAForInstructions
//
// Issue: dyn_cast assertion failure
// Location: Casting.h:644
// Error: Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"`
//
// When initializing aggregate members in a constructor with template parameters,
// CIR attempts an invalid cast operation.

struct a {
  template <typename b, typename c> a(b, c);
};
class d {
  a e;

public:
  d(int) : e(0, 0) {}
};
void f() { static d g(0); }
