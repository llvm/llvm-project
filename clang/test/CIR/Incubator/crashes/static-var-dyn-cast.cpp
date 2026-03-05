// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// dyn_cast on non-existent value - assertion failure
// Location: Casting.h:644
//
// Original failure: assertion_dyncast from LLVM build
// Reduced from /tmp/FormattedStream-a19c5f.cpp

struct a {
  template <typename b, typename c> a(b, c);
};
class d {
  a e;

public:
  d(int) : e(0, 0) {}
};
void f() { static d g(0); }
