// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// SD_Automatic storage duration for reference temporaries not implemented
// Location: CIRGenExpr.cpp:2356

struct S {
  S();
  ~S();
};

S create();

void f() {
  auto&& s = create();
}
