// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Constant expression NYI
// Location: CIRGenExprConst.cpp:1006
//
// Original failure: exprconst_1006 from LLVM build
// Reduced from /tmp/HexagonAttributeParser-40f1ed.cpp

class a {
public:
  int b(unsigned);
};
class c : a {
  struct d {
    int (c::*e)(unsigned);
  } static const f[];
};
const c::d c::f[]{&a::b};
