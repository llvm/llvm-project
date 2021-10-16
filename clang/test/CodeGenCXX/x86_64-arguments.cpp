// RUN: %clang_cc1 -disable-noundef-analysis -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Basic base class test.
struct f0_s0 { unsigned a; };
struct f0_s1 : public f0_s0 { void *b; };
// CHECK-LABEL: define{{.*}} void @_Z2f05f0_s1(i32 %a0.coerce0, i8* %a0.coerce1)
void f0(f0_s1 a0) { }

// Check with two eight-bytes in base class.
struct f1_s0 { unsigned a; unsigned b; float c; };
struct f1_s1 : public f1_s0 { float d;};
// CHECK-LABEL: define{{.*}} void @_Z2f15f1_s1(i64 %a0.coerce0, <2 x float> %a0.coerce1)
void f1(f1_s1 a0) { }

// Check with two eight-bytes in base class and merge.
struct f2_s0 { unsigned a; unsigned b; float c; };
struct f2_s1 : public f2_s0 { char d;};
// CHECK-LABEL: define{{.*}} void @_Z2f25f2_s1(i64 %a0.coerce0, i64 %a0.coerce1)
void f2(f2_s1 a0) { }

// PR5831
// CHECK-LABEL: define{{.*}} void @_Z2f34s3_1(i64 %x.coerce)
struct s3_0 {};
struct s3_1 { struct s3_0 a; long b; };
void f3(struct s3_1 x) {}

// CHECK-LABEL: define{{.*}} i64 @_Z4f4_0M2s4i(i64 %a)
// CHECK: define {{.*}} @_Z4f4_1M2s4FivE(i64 %a.coerce0, i64 %a.coerce1)
struct s4 {};
typedef int s4::* s4_mdp;
typedef int (s4::*s4_mfp)();
s4_mdp f4_0(s4_mdp a) { return a; }
s4_mfp f4_1(s4_mfp a) { return a; }

// A struct with <= one eightbyte before a member data pointer should still
// be allowed in registers.
// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mdp{{.*}}(i8* %a.coerce0, i64 %a.coerce1)
struct struct_with_mdp { char *a; s4_mdp b; };
void f_struct_with_mdp(struct_with_mdp a) { (void)a; }

// A struct with anything before a member function will be too big and
// goes in memory.
// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mfp_0{{.*}}(%struct{{.*}} byval(%struct{{.*}}) align 8 %a)
struct struct_with_mfp_0 { char a; s4_mfp b; };
void f_struct_with_mfp_0(struct_with_mfp_0 a) { (void)a; }

// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mfp_1{{.*}}(%struct{{.*}} byval(%struct{{.*}}) align 8 %a)
struct struct_with_mfp_1 { void *a; s4_mfp b; };
void f_struct_with_mfp_1(struct_with_mfp_1 a) { (void)a; }

namespace PR7523 {
struct StringRef {
  char *a;
};

void AddKeyword(StringRef, int x);

void foo() {
  // CHECK-LABEL: define{{.*}} void @_ZN6PR75233fooEv()
  // CHECK: call void @_ZN6PR752310AddKeywordENS_9StringRefEi(i8* {{.*}}, i32 4)
  AddKeyword(StringRef(), 4);
}
}

namespace PR7742 { // Also rdar://8250764
  struct s2 {
    float a[2];
  };

  struct c2 : public s2 {};

  // CHECK-LABEL: define{{.*}} <2 x float> @_ZN6PR77423fooEPNS_2c2E(%"struct.PR7742::c2"* %P)
  c2 foo(c2 *P) {
    return c2();
  }

}

namespace PR5179 {
  struct B {};

  struct B1 : B {
    int* pa;
  };

  struct B2 : B {
    B1 b1;
  };

  // CHECK-LABEL: define{{.*}} i8* @_ZN6PR51793barENS_2B2E(i32* %b2.coerce)
  const void *bar(B2 b2) {
    return b2.b1.pa;
  }
}

namespace test5 {
  struct Xbase { };
  struct Empty { };
  struct Y;
  struct X : public Xbase {
    Empty empty;
    Y f();
  };
  struct Y : public X {
    Empty empty;
  };
  X getX();
  int takeY(const Y&, int y);
  void g() {
    // rdar://8340348 - The temporary for the X object needs to have a defined
    // address when passed into X::f as 'this'.
    takeY(getX().f(), 42);
  }
  // CHECK: void @_ZN5test51gEv()
  // CHECK: alloca %"struct.test5::Y"
  // CHECK: alloca %"struct.test5::X"
  // CHECK: alloca %"struct.test5::Y"
}


// rdar://8360877
namespace test6 {
  struct outer {
    int x;
    struct epsilon_matcher {} e;
    int f;
  };

  int test(outer x) {
    return x.x + x.f;
  }
  // CHECK-LABEL: define{{.*}} i32 @_ZN5test64testENS_5outerE(i64 %x.coerce0, i32 %x.coerce1)
}

namespace test7 {
  struct StringRef {char* ptr; long len; };
  class A { public: ~A(); };
  A x(A, A, long, long, StringRef) { return A(); }
  // Check that the StringRef is passed byval instead of expanded
  // (which would split it between registers and memory).
  // rdar://problem/9686430
  // CHECK: define{{.*}} void @_ZN5test71xENS_1AES0_llNS_9StringRefE({{.*}} byval({{.*}}) align 8 {{%.*}})

  // And a couple extra related tests:
  A y(A, long double, long, long, StringRef) { return A(); }
  // CHECK: define{{.*}} void @_ZN5test71yENS_1AEellNS_9StringRefE({{.*}} i8*
  struct StringDouble {char * ptr; double d;};
  A z(A, A, A, A, A, StringDouble) { return A(); }
  A zz(A, A, A, A, StringDouble) { return A(); }
  // CHECK: define{{.*}} void @_ZN5test71zENS_1AES0_S0_S0_S0_NS_12StringDoubleE({{.*}} byval({{.*}}) align 8 {{%.*}})
  // CHECK: define{{.*}} void @_ZN5test72zzENS_1AES0_S0_S0_NS_12StringDoubleE({{.*}} i8*
}

namespace test8 {
  // CHECK: declare void @_ZN5test83fooENS_1BE(%"class.test8::B"* byval(%"class.test8::B") align 8)
  class A {
   char big[17];
  };

  class B : public A {};

  void foo(B b);
  void bar() {
   B b;
   foo(b);
  }
}

// PR4242
namespace test9 {
  // Large enough to be passed indirectly.
  struct S { void *data[3]; };

  struct T { void *data[2]; };

  // CHECK: define{{.*}} void @_ZN5test93fooEPNS_1SEPNS_1TE([[S:%.*]]* %0, [[T:%.*]]* %1)
  void foo(S*, T*) {}

  // CHECK: define{{.*}} void @_ZN5test91aEiiiiNS_1TEPv([[S]]* noalias sret([[S]]) align 8 {{%.*}}, i32 %0, i32 %1, i32 %2, i32 %3, [[T]]* byval([[T]]) align 8 %4, i8* %5)
  S a(int, int, int, int, T, void*) {
    return S();
  }

  // CHECK: define{{.*}} [[S]]* @_ZN5test91bEPNS_1SEiiiiNS_1TEPv([[S]]* {{%.*}}, i32 %0, i32 %1, i32 %2, i32 %3, [[T:%.*]]* byval([[T]]) align 8 %4, i8* %5)
  S* b(S* sret, int, int, int, int, T, void*) {
    return sret;
  }

  // CHECK: define{{.*}} void @_ZN5test91cEiiiNS_1TEPv([[S]]* noalias sret([[S]]) align 8 {{%.*}}, i32 %0, i32 %1, i32 %2, i8* {{%.*}}, i8* {{%.*}}, i8* %3)
  S c(int, int, int, T, void*) {
    return S();
  }

  // CHECK: define{{.*}} [[S]]* @_ZN5test91dEPNS_1SEiiiNS_1TEPv([[S]]* {{%.*}}, i32 %0, i32 %1, i32 %2, i8* {{%.*}}, i8* {{%.*}}, i8* %3)
  S* d(S* sret, int, int, int, T, void*) {
    return sret;
  }
}

namespace test10 {
#pragma pack(1)
struct BasePacked {
  char one;
  short two;
};
#pragma pack()
struct DerivedPacked : public BasePacked {
  int three;
};
// CHECK-LABEL: define{{.*}} i32 @_ZN6test1020FuncForDerivedPackedENS_13DerivedPackedE({{.*}}* byval({{.*}}) align 8
int FuncForDerivedPacked(DerivedPacked d) {
  return d.three;
}
}

namespace test11 {
union U {
  float f1;
  char __attribute__((__vector_size__(1))) f2;
};
int f(union U u) { return u.f2[1]; }
// CHECK-LABEL: define{{.*}} i32 @_ZN6test111fENS_1UE(i32
}
