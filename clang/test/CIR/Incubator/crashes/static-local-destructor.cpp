// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Crash from LLVM build with ClangIR
//
// Issue: Static local variable with destructor
// Location: CIRGenDecl.cpp:616
// Error: UNREACHABLE: NYI
//
// When a static local variable has a non-trivial destructor, CIR must
// register the destructor to run at program exit. This is not yet implemented.

class a {
public:
  ~a();
};
void b() { static a c; }
