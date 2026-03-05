// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Declaration handling NYI
// Location: CIRGenDecl.cpp:616
//
// Original failure: decl_616 from LLVM build
// Reduced from /tmp/MSFError-102e4d.cpp

class a {
public:
  ~a();
};
void b() { static a c; }
