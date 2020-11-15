// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++20 %s -triple x86_64-windows -emit-llvm -o - | FileCheck %s --check-prefix=MSABI

#define fold(x) (__builtin_constant_p(x) ? (x) : (x))

struct A { int a; const int b; };
template<A> void f() {}

// CHECK: define weak_odr void @_Z1fIXtl1ALi1ELi2EEEEvv(
// MSABI: define {{.*}} @"??$f@$2UA@@H00$$CBH01@@@YAXXZ"
template void f<A{1, 2}>();

struct B { const int *p; int k; };
template<B> void f() {}

int n = 0;
// CHECK: define weak_odr void @_Z1fIXtl1BadL_Z1nEEEEvv(
// MSABI: define {{.*}} @"??$f@$2UB@@PEBH1?n@@3HAH0A@@@@YAXXZ"
template void f<B{&n}>();
// CHECK: define weak_odr void @_Z1fIXtl1BLPKi0ELi1EEEEvv(
// MSABI: define {{.*}} @"??$f@$2UB@@PEBH0A@H00@@@YAXXZ"
template void f<B{nullptr, 1}>();
// CHECK: define weak_odr void @_Z1fIXtl1BEEEvv(
// MSABI: define {{.*}} @"??$f@$2UB@@PEBH0A@H0A@@@@YAXXZ"
template void f<B{nullptr}>();
// These are extensions, but they seem like the obvious manglings.
// CHECK: define weak_odr void @_Z1fIXtl1BLPKi32EEEEvv(
// MSABI: define {{.*}} @"??$f@$2UB@@PEBH0CA@H0A@@@@YAXXZ"
template void f<B{fold((int*)32)}>();
#ifndef _WIN32
// FIXME: On MS ABI, we mangle this the same as nullptr, despite considering a
// null pointer and zero bitcast to a pointer to be distinct pointer values.
// CHECK: define weak_odr void @_Z1fIXtl1BrcPKiLi0EEEEvv(
template void f<B{fold(reinterpret_cast<int*>(0))}>();
#endif

// Pointers to subobjects.
struct Nested { union { int k; int arr[2]; }; } nested[2];
struct Derived : A, Nested { int z; } extern derived;
// CHECK: define weak_odr void @_Z1fIXtl1BadsoKiL_Z7derivedE16EEEEvv
// MSABI: define {{.*}} void @"??$f@$2UB@@PEBH56E?derived@@3UDerived@@Az@@@H0A@@@@YAXXZ"
template void f<B{&derived.z}>();
// FIXME: We don't know the MS ABI mangling for array subscripting and
// past-the-end pointers yet.
#ifndef _WIN32
// CHECK: define weak_odr void @_Z1fIXtl1BadsoKiL_Z6nestedE_EEEEvv
template void f<B{&nested[0].k}>();
// CHECK: define weak_odr void @_Z1fIXtl1BadsoKiL_Z6nestedE16_0pEEEEvv
template void f<B{&nested[1].arr[2]}>();
// CHECK: define weak_odr void @_Z1fIXtl1BadsoKiL_Z7derivedE8pEEEEvv
template void f<B{&derived.b + 1}>();
// CHECK: define weak_odr void @_Z1fIXtl1BcvPKiplcvPcadL_Z7derivedELl16EEEEvv
template void f<B{fold(&derived.b + 3)}>();
#endif

// References to subobjects.
struct BR { const int &r; };
template<BR> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl2BRsoKiL_Z7derivedE16EEEEvv
// MSABI: define {{.*}} void @"??$f@$2UBR@@AEBH6E?derived@@3UDerived@@Az@@@@@YAXXZ"
template void f<BR{derived.z}>();
// FIXME: We don't know the MS ABI mangling for array subscripting yet.
#ifndef _WIN32
// CHECK: define weak_odr void @_Z1fIXtl2BRsoKiL_Z6nestedE_EEEEvv
template void f<BR{nested[0].k}>();
// CHECK: define weak_odr void @_Z1fIXtl2BRsoKiL_Z6nestedE12_0EEEEvv
template void f<BR{nested[1].arr[1]}>();
// CHECK: define weak_odr void @_Z1fIXtl2BRsoKiL_Z7derivedE4EEEEvv
template void f<BR{derived.b}>();
// CHECK: define weak_odr void @_Z1fIXtl2BRdecvPKiplcvPcadL_Z7derivedELl16EEEEvv
template void f<BR{fold(*(&derived.b + 3))}>();
#endif

// Qualification conversions.
struct C { const int *p; };
template<C> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1CadsoKiL_Z7derivedE16EEEEvv
// MSABI: define {{.*}} void @"??$f@$2UC@@PEBH56E?derived@@3UDerived@@Az@@@@@@YAXXZ"
template void f<C{&derived.z}>();
#ifndef _WIN32
// CHECK: define weak_odr void @_Z1fIXtl1CadsoKiL_Z7derivedE4EEEEvv
template void f<C{&derived.b}>();
#endif

// Pointers to members.
struct D { const int Derived::*p; int k; };
template<D> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1DLM7DerivedKi0ELi1EEEEvv
// MSABI: define {{.*}} @"??$f@$2UD@@PERDerived@@H0?0H00@@@YAXXZ"
template void f<D{nullptr, 1}>();
// CHECK: define weak_odr void @_Z1fIXtl1DEEEvv
// MSABI: define {{.*}} @"??$f@$2UD@@PERDerived@@H0?0H0A@@@@YAXXZ"
template void f<D{nullptr}>();
// CHECK: define weak_odr void @_Z1fIXtl1DadL_ZN7Derived1zEEEEEvv
// MSABI: define {{.*}} @"??$f@$2UD@@PERDerived@@H0BA@H0A@@@@YAXXZ"
template void f<D{&Derived::z}>();
#ifndef _WIN32
// CHECK: define weak_odr void @_Z1fIXtl1DmcM7DerivedKiadL_ZN1A1aEEEEEEvv
// MSABI-FIXME: define {{.*}} @"??$f@$2UD@@PERDerived@@H0A@H0A@@@@YAXXZ"
template void f<D{&A::a}>();
// CHECK: define weak_odr void @_Z1fIXtl1DmcM7DerivedKiadL_ZN1A1bEEEEEEvv
// MSABI-FIXME: define {{.*}} @"??$f@$2UD@@PERDerived@@H03H0A@@@@YAXXZ"
template void f<D{&A::b}>();
// FIXME: Is the Ut_1 mangling here correct?
// CHECK: define weak_odr void @_Z1fIXtl1DmcM7DerivedKiadL_ZN6NestedUt_1kEE8ELi2EEEEvv
// FIXME: This mangles the same as &A::a (bug in the MS ABI).
// MSABI-FIXME: define {{.*}} @"??$f@$2UD@@PERDerived@@H0A@H01@@@YAXXZ"
template void f<D{&Nested::k, 2}>();
struct MoreDerived : A, Derived { int z; };
// CHECK: define weak_odr void @_Z1fIXtl1DmcM7DerivedKiadL_ZN11MoreDerived1zEEn8EEEEvv
// MSABI-FIXME: define {{.*}} @"??$f@$2UD@@PERDerived@@H0BI@H0A@@@@YAXXZ"
template void f<D{(int Derived::*)&MoreDerived::z}>();
#endif

// FIXME: Pointers to member functions.

union E {
  int n;
  float f;
  constexpr E() {}
  constexpr E(int n) : n(n) {}
  constexpr E(float f) : f(f) {}
};
template<E> void f() {}

// Union members.
// CHECK: define weak_odr void @_Z1fIL1EEEvv(
// MSABI: define {{.*}} @"??$f@$7TE@@@@@YAXXZ"
template void f<E{}>();
// CHECK: define weak_odr void @_Z1fIXtl1EEEEvv(
// MSABI: define {{.*}} @"??$f@$7TE@@n@0A@@@@YAXXZ"
template void f<E(0)>();
// CHECK: define weak_odr void @_Z1fIXtl1Edi1nLi42EEEEvv(
// MSABI: define {{.*}} @"??$f@$7TE@@n@0CK@@@@YAXXZ"
template void f<E(42)>();
// CHECK: define weak_odr void @_Z1fIXtl1Edi1fLf00000000EEEEvv(
// MSABI: define {{.*}} @"??$f@$7TE@@0AA@@@@YAXXZ"
template void f<E(0.f)>();

// immintrin.h vector types.
typedef float __m128 __attribute__((__vector_size__(16)));
typedef double __m128d __attribute__((__vector_size__(16)));
typedef long long __m128i __attribute__((__vector_size__(16)));
struct M128 { __m128 a; };
struct M128D { __m128d b; };
struct M128I { __m128i c; };
template<M128> void f() {}
template<M128D> void f() {}
template<M128I> void f() {}
// MSABI: define {{.*}} @"??$f@$2UM128@@2T__m128@@3MADPIAAAAA@@AEAAAAAAA@@AEAEAAAAA@@AEAIAAAAA@@@@@@@YAXXZ"
template void f<M128{1, 2, 3, 4}>();
// MSABI: define {{.*}} @"??$f@$2UM128D@@2U__m128d@@3NBDPPAAAAAAAAAAAAA@@BEAAAAAAAAAAAAAAA@@@@@@@YAXXZ"
template void f<M128D{1, 2}>();
// FIXME: We define __m128i as a vector of long long, whereas MSVC appears to
// mangle it as if it were a vector of char.
// MSABI-FIXME: define {{.*}} @"??$f@$2UM128I@@2T__m128i@@3D00@01@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@0A@@@@@@@YAXXZ"
// MSABI: define {{.*}} @"??$f@$2UM128I@@2T__m128i@@3_J00@01@@@@@@YAXXZ"
template void f<M128I{1, 2}>();

// Extensions, and dropping trailing zero-initialized elements of 'tl'
// manglings.
typedef int __attribute__((ext_vector_type(3))) VI3;
struct F { VI3 v; _Complex int ci; _Complex float cf; };
template<F> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2ELi3EEtlCiLi4ELi5EEtlCfLf40c00000ELf40e00000EEEEEvv
// MSABI: define {{.*}} @"??$f@$2UF@@2T?$__vector@H$02@__clang@@3H00@01@02@@@2U?$_Complex@H@3@0304@2U?$_Complex@M@3@AEAMAAAAA@AEAOAAAAA@@@@@YAXXZ"
template void f<F{{1, 2, 3}, {4, 5}, {6, 7}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2ELi3EEtlCiLi4ELi5EEtlCfLf40c00000EEEEEvv
template void f<F{{1, 2, 3}, {4, 5}, {6, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2ELi3EEtlCiLi4ELi5EEEEEvv
template void f<F{{1, 2, 3}, {4, 5}, {0, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2ELi3EEtlCiLi4EEEEEvv
template void f<F{{1, 2, 3}, {4, 0}, {0, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2ELi3EEEEEvv
template void f<F{{1, 2, 3}, {0, 0}, {0, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1ELi2EEEEEvv
template void f<F{{1, 2, 0}, {0, 0}, {0, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FtlDv3_iLi1EEEEEvv
template void f<F{{1, 0, 0}, {0, 0}, {0, 0}}>();
// CHECK: define weak_odr void @_Z1fIXtl1FEEEvv
// MSABI: define {{.*}} @"??$f@$2UF@@2T?$__vector@H$02@__clang@@3H0A@@0A@@0A@@@@2U?$_Complex@H@3@0A@0A@@2U?$_Complex@M@3@AA@AA@@@@@YAXXZ"
template void f<F{{0, 0, 0}, {0, 0}, {0, 0}}>();

// Unnamed bit-fields.
struct G {
  int : 3;
  int a : 4;
  int : 5;
  int b : 6;
  int : 7;
};
template<G> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1GEEEvv
// MSABI: define {{.*}} @"??$f@$2UG@@H0A@H0A@@@@YAXXZ"
template void f<(G())>();
// CHECK: define weak_odr void @_Z1fIXtl1GLi1EEEEvv
// MSABI: define {{.*}} @"??$f@$2UG@@H00H0A@@@@YAXXZ"
template void f<G{1}>();
// CHECK: define weak_odr void @_Z1fIXtl1GLi1ELi2EEEEvv
// MSABI: define {{.*}} @"??$f@$2UG@@H00H01@@@YAXXZ"
template void f<G{1, 2}>();
// CHECK: define weak_odr void @_Z1fIXtl1GLin8ELin32EEEEvv
// MSABI: define {{.*}} @"??$f@$2UG@@H0?7H0?CA@@@@YAXXZ"
template void f<G{-8, -32}>();

// Empty and nearly-empty unions.
// Some of the MSVC manglings here are our invention, because MSVC rejects, but
// seem likely to be right.
union H1 {};
union H2 { int : 1, : 2, : 3; };
union H3 { int : 1, a, : 2, b, : 3; };
struct H4 { H2 h2; };
template<H1> void f() {}
template<H2> void f() {}
template<H3> void f() {}
template<H4> void f() {}
// CHECK: define weak_odr void @_Z1fIL2H1EEvv
// MSABI: define {{.*}} @"??$f@$7TH1@@@@@YAXXZ"
template void f<H1{}>();
// CHECK: define weak_odr void @_Z1fIL2H2EEvv
// MSABI: define {{.*}} @"??$f@$7TH2@@@@@YAXXZ"
template void f<H2{}>();
// CHECK: define weak_odr void @_Z1fIXtl2H3EEEvv
// MSABI: define {{.*}} @"??$f@$7TH3@@a@0A@@@@YAXXZ"
template void f<H3{.a = 0}>();
// CHECK: define weak_odr void @_Z1fIXtl2H3di1aLi1EEEEvv
// MSABI: define {{.*}} @"??$f@$7TH3@@a@00@@@YAXXZ"
template void f<H3{.a = 1}>();
// CHECK: define weak_odr void @_Z1fIXtl2H3di1bLi0EEEEvv
// MSABI: define {{.*}} @"??$f@$7TH3@@b@0A@@@@YAXXZ"
template void f<H3{.b = 0}>();
// CHECK: define weak_odr void @_Z1fIXtl2H4EEEvv
// MSABI: define {{.*}} @"??$f@$2UH4@@7TH2@@@@@@YAXXZ"
template void f<H4{}>();

// Floating-point.
struct I {
  float f;
  double d;
  long double ld;
};
template<I> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1IEEEvv
// MSABI: define {{.*}} @"??$f@$2UI@@MAA@NBA@OBA@@@@YAXXZ"
template void f<I{0.0, 0.0, 0.0}>();
// CHECK: define weak_odr void @_Z1fIXtl1ILf80000000ELd8000000000000000ELe80000000000000000000EEEEvv
// MSABI: define {{.*}} @"??$f@$2UI@@MAIAAAAAAA@NBIAAAAAAAAAAAAAAA@OBIAAAAAAAAAAAAAAA@@@@YAXXZ"
template void f<I{-0.0, -0.0, -0.0}>();
// CHECK: define weak_odr void @_Z1fIXtl1ILf3f800000ELd4000000000000000ELec000c000000000000000EEEEvv
// MSABI: define {{.*}} @"??$f@$2UI@@MADPIAAAAA@NBEAAAAAAAAAAAAAAA@OBMAAIAAAAAAAAAAAA@@@@YAXXZ"
template void f<I{1.0, 2.0, -3.0}>();
// CHECK: define {{.*}} @_Z1fIXtl1ILf00000000ELd0000000000000000ELe3bcd8000000000000000EEEEvv
// Note that "small integer" special-case manglings 'A@', '0', '1', ... are
// used here and represent tiny denormal values!
// MSABI: define {{.*}} @"??$f@$2UI@@MAA@NBA@OB0@@@YAXXZ"
template void f<I{0.0, 0.2e-323, 0.5e-323}>();
// CHECK: define {{.*}} @_Z1fIXtl1ILf00000000ELd0000000000000002ELebbce8000000000000000EEEEvv
// ... but the special-case '?' mangling for bit 63 being set is not used.
// MSABI: define {{.*}} @"??$f@$2UI@@MAA@NB1OBIAAAAAAAAAAAAAAC@@@@YAXXZ"
template void f<I{0.0, 1.0e-323, -1.0e-323}>();

// Base classes and members of class type.
struct J1 { int a, b; };
struct JEmpty {};
struct J2 { int c, d; };
struct J : J1, JEmpty, J2 { int e; };
template<J> void f() {}
// CHECK: define weak_odr void @_Z1fIXtl1JEEEvv
// MSABI: define {{.*}} @"??$f@$2UJ@@2UJ1@@H0A@H0A@@2UJEmpty@@@2UJ2@@H0A@H0A@@H0A@@@@YAXXZ"
template void f<J{}>();
// CHECK: define weak_odr void @_Z1fIXtl1Jtl2J1Li1ELi2EEtl6JEmptyEtl2J2Li3ELi4EELi5EEEEvv
// MSABI: define {{.*}} @"??$f@$2UJ@@2UJ1@@H00H01@2UJEmpty@@@2UJ2@@H02H03@H04@@@YAXXZ"
template void f<J{{1, 2}, {}, {3, 4}, 5}>();

struct J3 { J1 j1; };
template<J3> void f() {}
// CHECK: define {{.*}} @_Z1fIXtl2J3tl2J1Li1ELi2EEEEEvv
// MSABI: define {{.*}} @"??$f@$2UJ3@@2UJ1@@H00H01@@@@YAXXZ"
template void f<J3{1, 2}>();

// Arrays.
struct K { int n[2][3]; };
template<K> void f() {}
// CHECK: define {{.*}} @_Z1fIXtl1KtlA2_A3_itlS1_Li1ELi2EEEEEEvv
// MSABI: define {{.*}} @"??$f@$2UK@@3$$BY02H3H00@01@0A@@@@3H0A@@0A@@0A@@@@@@@@YAXXZ"
template void f<K{1, 2}>();
// CHECK: define {{.*}} @_Z1fIXtl1KtlA2_A3_itlS1_Li1ELi2ELi3EEtlS1_Li4ELi5ELi6EEEEEEvv
// MSABI: define {{.*}} @"??$f@$2UK@@3$$BY02H3H00@01@02@@@3H03@04@05@@@@@@@YAXXZ"
template void f<K{1, 2, 3, 4, 5, 6}>();

struct K1 { int a, b; };
struct K2 : K1 { int c; };
struct K3 { K2 k2[2]; };
template<K3> void f() {}
// CHECK: define {{.*}} @_Z1fIXtl2K3tlA2_2K2tlS1_tl2K1Li1EEEEEEEvv
// MSABI: define {{.*}} @"??$f@$2UK3@@3UK2@@2U2@2UK1@@H00H0A@@H0A@@@2U2@2U3@H0A@H0A@@H0A@@@@@@@YAXXZ"
template void f<K3{1}>();
template void f<K3{1, 2, 3, 4, 5, 6}>();

namespace CvQualifiers {
  struct A { const int a; int *const b; int c; };
  template<A> void f() {}
  // CHECK: define {{.*}} @_ZN12CvQualifiers1fIXtlNS_1AELi0ELPi0ELi1EEEEEvv
  // MSABI: define {{.*}} @"??$f@$2UA@CvQualifiers@@$$CBH0A@QEAH0A@H00@@CvQualifiers@@YAXXZ"
  template void f<A{.c = 1}>();

  using T1 = const int;
  using T2 = T1[5];
  using T3 = const T2;
  struct B { T3 arr; };
  template<B> void f() {}
  // CHECK: define {{.*}} @_ZN12CvQualifiers1fIXtlNS_1BEtlA5_iLi1ELi2ELi3ELi4ELi5EEEEEEvv
  // MSABI: define {{.*}} @"??$f@$2UB@CvQualifiers@@3$$CBH00@01@02@03@04@@@@CvQualifiers@@YAXXZ"
  template void f<B{1, 2, 3, 4, 5}>();
}

struct L {
  signed char a = -1;
  unsigned char b = -1;
  short c = -1;
  unsigned short d = -1;
  int e = -1;
  unsigned int f = -1;
  long g = -1;
  unsigned long h = -1;
  long long i = -1;
  unsigned long long j = -1;
};
template<L> void f() {}
// CHECK: define {{.*}} @_Z1fIXtl1LLan1ELh255ELsn1ELt65535ELin1ELj4294967295ELln1ELm18446744073709551615ELxn1ELy18446744073709551615EEEEvv
// MSABI: define {{.*}} @"??$f@$2UL@@C0?0E0PP@F0?0G0PPPP@H0?0I0PPPPPPPP@J0?0K0PPPPPPPP@_J0?0_K0?0@@@YAXXZ"
template void f<L{}>();

// Template parameter objects.
struct M { int n; };
template<M a> constexpr const M &f() { return a; }
// CHECK: define {{.*}} @_Z1fIXtl1MLi42EEEERKS0_v
// CHECK: ret {{.*}} @_ZTAXtl1MLi42EEE
// MSABI: define {{.*}} @"??$f@$2UM@@H0CK@@@@YAAEBUM@@XZ"
// MSABI: ret {{.*}} @"??__N2UM@@H0CK@@@"
template const M &f<M{42}>();

template<const M *p> void g() {}
// CHECK: define {{.*}} @_Z1gIXadL_ZTAXtl1MLi10EEEEEEvv
// MSABI: define {{.*}} @"??$g@$1??__N2UM@@H09@@@@YAXXZ"
template void g<&f<M{10}>()>();
