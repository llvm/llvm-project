// RUN: %clang_cc1 -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++20 -include-pch %t -ftime-trace=%t.json -ftime-trace-granularity=0 -fsyntax-only %s

// expected-no-diagnostics

#ifndef HEADER_INCLUDED
#define HEADER_INCLUDED

inline namespace {

// The first declarations give f's body references to many function templates.
#define DECLARE_G(N) template <typename T> T g##N(T v) { return v; }

DECLARE_G(0)
DECLARE_G(1)
DECLARE_G(2)
DECLARE_G(3)
DECLARE_G(4)
DECLARE_G(5)
DECLARE_G(6)
DECLARE_G(7)
DECLARE_G(8)
DECLARE_G(9)
DECLARE_G(10)
DECLARE_G(11)
DECLARE_G(12)
DECLARE_G(13)
DECLARE_G(14)
DECLARE_G(15)
DECLARE_G(16)
DECLARE_G(17)
DECLARE_G(18)
DECLARE_G(19)
DECLARE_G(20)
DECLARE_G(21)
DECLARE_G(22)
DECLARE_G(23)
DECLARE_G(24)
DECLARE_G(25)
DECLARE_G(26)
DECLARE_G(27)
DECLARE_G(28)
DECLARE_G(29)
DECLARE_G(30)
DECLARE_G(31)

#undef DECLARE_G

template <typename T> T f(T v) {
  return g0(v) + g1(v) + g2(v) + g3(v) + g4(v) + g5(v) + g6(v) +
         g7(v) + g8(v) + g9(v) + g10(v) + g11(v) + g12(v) + g13(v) +
         g14(v) + g15(v) + g16(v) + g17(v) + g18(v) + g19(v) + g20(v) +
         g21(v) + g22(v) + g23(v) + g24(v) + g25(v) + g26(v) + g27(v) +
         g28(v) + g29(v) + g30(v) + g31(v);
}

// These later declarations are deserialized while -ftime-trace prints the
// qualified name of a specialization lookup. Loading enough of them grows the
// ASTReader specialization DenseMap and used to invalidate the active lookup.
#define DECLARE_G(N) template <typename T> T g##N();

DECLARE_G(0)
DECLARE_G(1)
DECLARE_G(2)
DECLARE_G(3)
DECLARE_G(4)
DECLARE_G(5)
DECLARE_G(6)
DECLARE_G(7)
DECLARE_G(8)
DECLARE_G(9)
DECLARE_G(10)
DECLARE_G(11)
DECLARE_G(12)
DECLARE_G(13)
DECLARE_G(14)
DECLARE_G(15)
DECLARE_G(16)
DECLARE_G(17)
DECLARE_G(18)
DECLARE_G(19)
DECLARE_G(20)
DECLARE_G(21)
DECLARE_G(22)
DECLARE_G(23)
DECLARE_G(24)
DECLARE_G(25)
DECLARE_G(26)
DECLARE_G(27)
DECLARE_G(28)
DECLARE_G(29)
DECLARE_G(30)
DECLARE_G(31)

#undef DECLARE_G

} // namespace

#else

int x;
void i() { f(x); }

#endif
