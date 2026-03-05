// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Bitfield bool to int conversion - type cast assertion failure
// Location: Casting.h:560

struct a {
  bool b : 1;
};
class c {
public:
  void operator<<(int);
};
void d(c e, a f) { e << f.b; }
