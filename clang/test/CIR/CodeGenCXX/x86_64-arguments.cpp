// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-unknown-unknown -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-unknown-unknown -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-unknown-unknown -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// All cases from CodeGenCXX/x86_64-arguments.cpp (Itanium x86_64 only).
// Tests C++ struct passing with inheritance, member pointers, empty bases,
// packed structs, non-trivially-destructible types, and unions.

// Basic base class test.
struct f0_s0 { unsigned a; };
struct f0_s1 : public f0_s0 { void *b; };
void f0(f0_s1 a0) { }

// CIR-LABEL: cir.func {{.*}} @_Z2f05f0_s1
// LLVM-LABEL: define {{.*}} void @_Z2f05f0_s1(
// OGCG-LABEL: define {{.*}} void @_Z2f05f0_s1(i32 %a0.coerce0, ptr %a0.coerce1)

// Two eight-bytes in base class.
struct f1_s0 { unsigned a; unsigned b; float c; };
struct f1_s1 : public f1_s0 { float d;};
void f1(f1_s1 a0) { }

// CIR-LABEL: cir.func {{.*}} @_Z2f15f1_s1
// LLVM-LABEL: define {{.*}} void @_Z2f15f1_s1(
// OGCG-LABEL: define {{.*}} void @_Z2f15f1_s1(i64 %a0.coerce0, <2 x float> %a0.coerce1)

// Two eight-bytes in base class and merge.
struct f2_s0 { unsigned a; unsigned b; float c; };
struct f2_s1 : public f2_s0 { char d;};
void f2(f2_s1 a0) { }

// CIR-LABEL: cir.func {{.*}} @_Z2f25f2_s1
// LLVM-LABEL: define {{.*}} void @_Z2f25f2_s1(
// OGCG-LABEL: define {{.*}} void @_Z2f25f2_s1(i64 %a0.coerce0, i64 %a0.coerce1)

// PR5831
struct s3_0 {};
struct s3_1 { struct s3_0 a; long b; };
void f3(struct s3_1 x) {}

// CIR-LABEL: cir.func {{.*}} @_Z2f34s3_1
// LLVM-LABEL: define {{.*}} void @_Z2f34s3_1(
// OGCG-LABEL: define {{.*}} void @_Z2f34s3_1(i64 %x.coerce)

// Member data pointer and member function pointer.
struct s4 {};
typedef int s4::* s4_mdp;
typedef int (s4::*s4_mfp)();
s4_mdp f4_0(s4_mdp a) { return a; }
s4_mfp f4_1(s4_mfp a) { return a; }

// CIR-LABEL: cir.func {{.*}} @_Z4f4_0M2s4i
// CIR-LABEL: cir.func {{.*}} @_Z4f4_1M2s4FivE

// LLVM-LABEL: define {{.*}} i64 @_Z4f4_0M2s4i(i64
// LLVM-LABEL: define {{.*}} @_Z4f4_1M2s4FivE(

// OGCG-LABEL: define {{.*}} i64 @_Z4f4_0M2s4i(i64 %a)
// OGCG: define {{.*}} @_Z4f4_1M2s4FivE(i64 %a.coerce0, i64 %a.coerce1)

// Struct with member data pointer (fits in registers).
struct struct_with_mdp { char *a; s4_mdp b; };
void f_struct_with_mdp(struct_with_mdp a) { (void)a; }

// CIR-LABEL: cir.func {{.*}} @_Z17f_struct_with_mdp
// LLVM-LABEL: define {{.*}} void @_Z17f_struct_with_mdp
// OGCG-LABEL: define {{.*}} void @{{.*}}f_struct_with_mdp{{.*}}(ptr %a.coerce0, i64 %a.coerce1)

// Struct with member function pointer (too big, goes to memory).
struct struct_with_mfp_0 { char a; s4_mfp b; };
void f_struct_with_mfp_0(struct_with_mfp_0 a) { (void)a; }

// CIR-LABEL: cir.func {{.*}} @_Z19f_struct_with_mfp_0
// LLVM-LABEL: define {{.*}} void @_Z19f_struct_with_mfp_0
// OGCG-LABEL: define {{.*}} void @{{.*}}f_struct_with_mfp_0{{.*}}(ptr byval(%struct{{.*}}) align 8 %a)

struct struct_with_mfp_1 { void *a; s4_mfp b; };
void f_struct_with_mfp_1(struct_with_mfp_1 a) { (void)a; }

// CIR-LABEL: cir.func {{.*}} @_Z19f_struct_with_mfp_1
// LLVM-LABEL: define {{.*}} void @_Z19f_struct_with_mfp_1
// OGCG-LABEL: define {{.*}} void @{{.*}}f_struct_with_mfp_1{{.*}}(ptr byval(%struct{{.*}}) align 8 %a)

namespace PR7523 {
struct StringRef { char *a; };
void AddKeyword(StringRef, int x);
void foo() {
  AddKeyword(StringRef(), 4);
}
}

// CIR-LABEL: cir.func {{.*}} @_ZN6PR75233fooEv
// CIR:   cir.call @_ZN6PR752310AddKeywordENS_9StringRefEi

// LLVM-LABEL: define {{.*}} void @_ZN6PR75233fooEv(
// LLVM:   call void @_ZN6PR752310AddKeywordENS_9StringRefEi(

// OGCG-LABEL: define {{.*}} void @_ZN6PR75233fooEv(
// OGCG:   call void @_ZN6PR752310AddKeywordENS_9StringRefEi(ptr {{.*}}, i32 4)

namespace PR7742 {
  struct s2 { float a[2]; };
  struct c2 : public s2 {};
  c2 foo(c2 *P) { return c2(); }
}

// CIR-LABEL: cir.func {{.*}} @_ZN6PR77423fooEPNS_2c2E
// LLVM-LABEL: define {{.*}} @_ZN6PR77423fooEPNS_2c2E(
// OGCG-LABEL: define {{.*}} <2 x float> @_ZN6PR77423fooEPNS_2c2E(ptr %P)

namespace PR5179 {
  struct B {};
  struct B1 : B { int* pa; };
  struct B2 : B { B1 b1; };
  const void *bar(B2 b2) { return b2.b1.pa; }
}

// CIR-LABEL: cir.func {{.*}} @_ZN6PR51793barENS_2B2E
// LLVM-LABEL: define {{.*}} @_ZN6PR51793barENS_2B2E(
// OGCG-LABEL: define {{.*}} ptr @_ZN6PR51793barENS_2B2E(ptr %b2.coerce)

namespace test5 {
  struct Xbase { };
  struct Empty { };
  struct Y;
  struct X : public Xbase { Empty empty; Y f(); };
  struct Y : public X { Empty empty; };
  X getX();
  int takeY(const Y&, int y);
  void g() { takeY(getX().f(), 42); }
}

// CIR-LABEL: cir.func {{.*}} @_ZN5test51gEv
// CIR:   cir.alloca !rec_{{.*}}Y
// CIR:   cir.alloca !rec_{{.*}}X
// CIR:   cir.call @_ZN5test54getXEv
// CIR:   cir.call @_ZN5test51X1fEv
// CIR:   cir.call @_ZN5test55takeYERKNS_1YEi

// LLVM-LABEL: define {{.*}} void @_ZN5test51gEv(
// LLVM:   alloca %"struct.test5::Y"
// LLVM:   alloca %"struct.test5::X"

// OGCG: void @_ZN5test51gEv()
// OGCG:   alloca %"struct.test5::Y"
// OGCG:   alloca %"struct.test5::X"
// OGCG:   alloca %"struct.test5::Y"

namespace test6 {
  struct outer { int x; struct epsilon_matcher {} e; int f; };
  int test(outer x) { return x.x + x.f; }
}

// CIR-LABEL: cir.func {{.*}} @_ZN5test64testENS_5outerE
// LLVM-LABEL: define {{.*}} i32 @_ZN5test64testENS_5outerE(
// OGCG-LABEL: define {{.*}} i32 @_ZN5test64testENS_5outerE(i64 %x.coerce0, i32 %x.coerce1)

namespace test7 {
  struct StringRef {char* ptr; long len; };
  class A { public: ~A(); };
  A x(A, A, long, long, StringRef) { return A(); }
  A y(A, long double, long, long, StringRef) { return A(); }
  struct StringDouble {char * ptr; double d;};
  A z(A, A, A, A, A, StringDouble) { return A(); }
  A zz(A, A, A, A, StringDouble) { return A(); }
}

// CIR-LABEL: cir.func {{.*}} @_ZN5test71xENS_1AES0_llNS_9StringRefE
// CIR-LABEL: cir.func {{.*}} @_ZN5test71yENS_1AEellNS_9StringRefE
// CIR-LABEL: cir.func {{.*}} @_ZN5test71zENS_1AES0_S0_S0_S0_NS_12StringDoubleE
// CIR-LABEL: cir.func {{.*}} @_ZN5test72zzENS_1AES0_S0_S0_NS_12StringDoubleE

// LLVM-LABEL: define {{.*}} @_ZN5test71xENS_1AES0_llNS_9StringRefE(
// LLVM-LABEL: define {{.*}} @_ZN5test71yENS_1AEellNS_9StringRefE(
// LLVM-LABEL: define {{.*}} @_ZN5test71zENS_1AES0_S0_S0_S0_NS_12StringDoubleE(
// LLVM-LABEL: define {{.*}} @_ZN5test72zzENS_1AES0_S0_S0_NS_12StringDoubleE(

// OGCG: define{{.*}} void @_ZN5test71xENS_1AES0_llNS_9StringRefE({{.*}} byval({{.*}}) align 8 {{%.*}})
// OGCG: define{{.*}} void @_ZN5test71yENS_1AEellNS_9StringRefE({{.*}} ptr
// OGCG: define{{.*}} void @_ZN5test71zENS_1AES0_S0_S0_S0_NS_12StringDoubleE({{.*}} byval({{.*}}) align 8 {{%.*}})
// OGCG: define{{.*}} void @_ZN5test72zzENS_1AES0_S0_S0_NS_12StringDoubleE({{.*}} ptr

namespace test8 {
  class A { char big[17]; };
  class B : public A {};
  void foo(B b);
  void bar() { B b; foo(b); }
}

// CIR-LABEL: cir.func {{.*}} @_ZN5test83barEv
// CIR:   cir.call @_ZN5test83fooENS_1BE

// LLVM-LABEL: define {{.*}} void @_ZN5test83barEv(
// LLVM:   call void @_ZN5test83fooENS_1BE(

// OGCG-LABEL: define {{.*}} void @_ZN5test83barEv(
// OGCG:   call void @_ZN5test83fooENS_1BE(ptr byval(%"class.test8::B") align 8

namespace test9 {
  struct S { void *data[3]; };
  struct T { void *data[2]; };
  void foo(S*, T*) {}
  S a(int, int, int, int, T, void*) { return S(); }
  S* b(S* sret, int, int, int, int, T, void*) { return sret; }
  S c(int, int, int, T, void*) { return S(); }
  S* d(S* sret, int, int, int, T, void*) { return sret; }
}

// CIR-LABEL: cir.func {{.*}} @_ZN5test93fooEPNS_1SEPNS_1TE
// CIR-LABEL: cir.func {{.*}} @_ZN5test91aEiiiiNS_1TEPv
// CIR-LABEL: cir.func {{.*}} @_ZN5test91bEPNS_1SEiiiiNS_1TEPv
// CIR-LABEL: cir.func {{.*}} @_ZN5test91cEiiiNS_1TEPv
// CIR-LABEL: cir.func {{.*}} @_ZN5test91dEPNS_1SEiiiNS_1TEPv

// LLVM-LABEL: define {{.*}} void @_ZN5test93fooEPNS_1SEPNS_1TE(ptr {{.*}}, ptr
// LLVM-LABEL: define {{.*}} @_ZN5test91aEiiiiNS_1TEPv(
// LLVM-LABEL: define {{.*}} ptr @_ZN5test91bEPNS_1SEiiiiNS_1TEPv(
// LLVM-LABEL: define {{.*}} @_ZN5test91cEiiiNS_1TEPv(
// LLVM-LABEL: define {{.*}} ptr @_ZN5test91dEPNS_1SEiiiNS_1TEPv(

// OGCG: define{{.*}} void @_ZN5test93fooEPNS_1SEPNS_1TE(ptr %0, ptr %1)
// OGCG: define{{.*}} void @_ZN5test91aEiiiiNS_1TEPv(ptr dead_on_unwind noalias writable sret({{.*}}) align 8 {{%.*}}, i32 %0, i32 %1, i32 %2, i32 %3, ptr byval({{.*}}) align 8 %4, ptr %5)
// OGCG: define{{.*}} ptr @_ZN5test91bEPNS_1SEiiiiNS_1TEPv(ptr {{%.*}}, i32 %0, i32 %1, i32 %2, i32 %3, ptr byval({{.*}}) align 8 %4, ptr %5)
// OGCG: define{{.*}} void @_ZN5test91cEiiiNS_1TEPv(ptr dead_on_unwind noalias writable sret({{.*}}) align 8 {{%.*}}, i32 %0, i32 %1, i32 %2, ptr {{%.*}}, ptr {{%.*}}, ptr %3)
// OGCG: define{{.*}} ptr @_ZN5test91dEPNS_1SEiiiNS_1TEPv(ptr {{%.*}}, i32 %0, i32 %1, i32 %2, ptr {{%.*}}, ptr {{%.*}}, ptr %3)

namespace test10 {
#pragma pack(1)
struct BasePacked { char one; short two; };
#pragma pack()
struct DerivedPacked : public BasePacked { int three; };
int FuncForDerivedPacked(DerivedPacked d) { return d.three; }
}

// CIR-LABEL: cir.func {{.*}} @_ZN6test1020FuncForDerivedPackedENS_13DerivedPackedE
// LLVM-LABEL: define {{.*}} i32 @_ZN6test1020FuncForDerivedPackedENS_13DerivedPackedE(
// OGCG-LABEL: define {{.*}} i32 @_ZN6test1020FuncForDerivedPackedENS_13DerivedPackedE(ptr byval({{.*}}) align 8

namespace test11 {
union U {
  float f1;
  char __attribute__((__vector_size__(1))) f2;
};
int f(union U u) { return u.f2[1]; }
}

// CIR-LABEL: cir.func {{.*}} @_ZN6test111fENS_1UE
// LLVM-LABEL: define {{.*}} i32 @_ZN6test111fENS_1UE(
// OGCG-LABEL: define {{.*}} i32 @_ZN6test111fENS_1UE(i32
