// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// APValue emission not implemented - assertion failure
// Location: CIRGenExprConst.cpp:2075
//
// Original failure: assertion_apvalue from LLVM build
// Reduced from /tmp/MSP430AttributeParser-875bbc.cpp

template <typename a, int b> struct c {
  typedef a d[b];
};
template <typename a, int b> struct h {
  c<a, b>::d e;
};
enum f { g };
class i {
  struct m {
    f j;
    int (i::*k)(f);
  };
  static const h<m, 4> l;
  int n(f);
};
constexpr h<i::m, 4> i::l{g, &i::n, {}, {}, {}};
