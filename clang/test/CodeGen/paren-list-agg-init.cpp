// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

template <typename T>
struct IsChar {
  constexpr operator bool() const { return false; }
};

template<>
struct IsChar<char> {
  constexpr operator bool() const { return true; }
};

template <typename T>
concept SameAsChar = (bool)IsInt<T>();

// CHECK-DAG: [[STRUCT_A:%.*]] = type { i8, double }
struct A {
  char i;
  double j;

  template <SameAsChar T>
  operator T() const { return i; };
};

// CHECK-DAG: [[STRUCT_B:%.*]] = type { [[STRUCT_A]], i32 }
struct B {
  A a;
  int b;
};

// CHECK-DAG: [[STRUCT_C:%.*]] = type <{ [[STRUCT_B]], [[STRUCT_A]], i32, [4 x i8] }>
struct C : public B, public A {
  int c;
};

// CHECK-DAG: [[STRUCT_D:%.*]] = type { [[STRUCT_A]], [[STRUCT_A]], i8, [[STRUCT_A]] }
struct D {
  A a;
  A b = A{2, 2.0};
  unsigned : 2;
  A c;
};

// CHECK-DAG: [[STRUCT_E:%.*]] = type { i32, ptr }
struct E {
  int a;
  const char* fn = __builtin_FUNCTION();
  ~E() {};
};

// CHECK-DAG: [[STRUCT_F:%.*]] = type { i8 }
struct F {
  F (int i = 1);
  F (const F &f) = delete;
  F (F &&f) = default;
};

// CHECK-DAG: [[STRUCT_G:%.*]] = type <{ i32, [[STRUCT_F]], [3 x i8] }>
struct G {
  int a;
  F f;
};

// CHECK-DAG: [[UNION_U:%.*]] = type { [[STRUCT_A]] }
// CHECK-DAG: [[STR:@.*]] = private unnamed_addr constant [6 x i8] {{.*}}foo18{{.*}}, align 1
union U {
  unsigned : 1;
  A a;
  char b;
};


namespace gh61145 {
  // CHECK-DAG: [[STRUCT_VEC:%.*]] = type { i8 }
  struct Vec {
    Vec();
    Vec(Vec&&);
    ~Vec();
  };

  // CHECK-DAG: [[STRUCT_S1:%.*]] = type { [[STRUCT_VEC]] }
  struct S1 {
    Vec v;
  };

  // CHECK-DAG: [[STRUCT_S2:%.*]] = type { [[STRUCT_VEC]], i8 }
  struct S2 {
    Vec v;
    char c;
  };
}

namespace gh62266 {
  // CHECK-DAG: [[STRUCT_H:%.*H.*]] = type { i32, i32 }
  template <int J>
  struct H {
    int i;
    int j = J;
  };
}

namespace gh61567 {
  // CHECK-DAG: [[STRUCT_I:%.*I.*]] = type { i32, ptr }
  struct I {
    int a;
    int&& r = 2;
  };
}

// CHECK-DAG: [[A1:@.*a1.*]] = internal constant [[STRUCT_A]] { i8 3, double 2.000000e+00 }, align 8
constexpr A a1(3.1, 2.0);
// CHECK-DAG: [[A2:@.*a2.*]] = internal constant [[STRUCT_A]] { i8 99, double 0.000000e+00 }, align 8
constexpr auto a2 = static_cast<A>('c');
// CHECK-DAG: [[B1:@.*b1.*]] = internal constant [[STRUCT_B]] { [[STRUCT_A]] { i8 99, double 0.000000e+00 }, i32 0 }, align 8
constexpr B b1(A('c'));
// CHECK-DAG: [[C1:@.*c1.*]] = internal constant { [[STRUCT_A]], i32, [4 x i8], i8, double, i32 } { [[STRUCT_A]] { i8 99, double 0.000000e+00 }, i32 0, [4 x i8] undef, i8 3, double 2.000000e+00, i32 0 }, align
constexpr C c1(b1, a1);
// CHECK-DAG: [[U1:@.*u1.*]] = internal constant [[UNION_U]] { [[STRUCT_A]] { i8 1, double 1.000000e+00 } }, align 8
constexpr U u1(A(1, 1));
// CHECK-DAG: [[D1:@.*d1.*]] = internal constant { [[STRUCT_A]], [[STRUCT_A]], [8 x i8], [[STRUCT_A]] } { [[STRUCT_A]] { i8 2, double 2.000000e+00 }, [[STRUCT_A]] { i8 2, double 2.000000e+00 }, [8 x i8] undef, [[STRUCT_A]] zeroinitializer }, align 8
constexpr D d1(A(2, 2));
// CHECK-DAG: [[ARR1:@.*arr1.*]] = internal constant [3 x i32] [i32 1, i32 2, i32 0], align 4
constexpr int arr1[3](1, 2);
// CHECK-DAG: [[ARR4:@.*arr4.*]] = internal constant [1 x i32] [i32 1], align 4
constexpr int arr4[](1);
// CHECK-DAG: [[ARR5:@.*arr5.*]] = internal constant [2 x i32] [i32 2, i32 0], align 4
constexpr int arr5[2](2);

// CHECK: define dso_local { i8, double } @{{.*foo1.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[RETVAL:%.*]] = alloca [[STRUCT_A]], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[A1]], i64 16, i1 false)
// CHECK-NEXT: [[TMP_0:%.*]] = load { i8, double }, ptr [[RETVAL]], align 8
// CHECK-NEXT: ret { i8, double } [[TMP_0]]
A foo1() {
  return a1;
}

// CHECK: define dso_local void @{{.*foo2.*}}(ptr dead_on_unwind noalias writable sret([[STRUCT_B]]) align 8 [[AGG_RESULT:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[AGG_RESULT]], ptr align 8 [[B1]], i64 24, i1 false)
// CHECK-NEXT: ret void
B foo2() {
  return b1;
}

// CHECK: define dso_local void @{{.*foo3.*}}(ptr dead_on_unwind noalias writable sret([[STRUCT_C]]) align 8 [[AGG_RESULT:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[AGG_RESULT]], ptr align 8 [[C1]], i64 48, i1 false)
// CHECK-NEXT: ret void
C foo3() {
  return c1;
}

// CHECK: define dso_local void @{{.*foo4.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[C2:%.*]] = alloca [[STRUCT_C:%.*]], align 8
// CHECK-NEXT: [[REF_TMP:%.*]] = alloca [[STRUCT_B:%.*]], align 8
// CHECK-NEXT: [[REF_TMP_1:%.*]] = alloca [[STRUCT_A:%.*]], align 8
// CHECK-NEXT: [[A:%.*]] = getelementptr inbounds [[STRUCT_B]], ptr [[REF_TMP]], i32 0, i32 0
// CHECK-NEXT: [[I:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: store i8 1, ptr [[I]], align 8
// CHECK-NEXT: [[J:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// CHECK-NEXT: [[B:%.*]] = getelementptr inbounds [[STRUCT_B]], ptr [[REF_TMP]], i32 0, i32 1
// CHECK-NEXT: store i32 1, ptr [[B]], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[C2]], ptr align 8 [[REF_TMP]], i64 24, i1 false)
// CHECK-NEXT: [[TMP_0:%.*]] = getelementptr inbounds i8, ptr [[C2]], i64 24
// CHECK-NEXT: [[I2:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[REF_TMP_1]], i32 0, i32 0
// CHECK-NEXT: store i8 97, ptr [[I2]], align 8
// CHECK-NEXT: [[J3:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[REF_TMP_1]], i32 0, i32 1
// CHECK-NEXT: store double 0.000000e+00, ptr [[J3]], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[TMP_0]], ptr align 8 [[REF_TMP_1]], i64 16, i1 false)
// CHECK-NEXT: [[C:%.*]] = getelementptr inbounds %struct.C, ptr %c2, i32 0, i32 2
// CHECK-NEXT: store i32 2, ptr %c, align 8
// CHECK-NEXT: ret void
void foo4() {
  C c2(B(A(1, 1), 1), A('a'), 2);
}

// CHECK: define dso_local { i64, double } @{{.*foo5.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT [[RETVAL:%.*]] = alloca [[UNION_U]], align 8
// CHECK-NEXT call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[U1]], i64 16, i1 false)
// CHECK-NEXT [[COERCE_DIVE:%.*]] = getelementptr inbounds [[UNION_U]], ptr %retval, i32 0, i32 0
// CHECK-NEXT [[TMP_0:%.*]] = load { i64, double }, ptr [[COERCE_DIVE]], align 8
// CHECK-NEXT ret { i64, double } [[TMP_0]]
U foo5() {
  return u1;
}


// CHECK: define dso_local { i64, double } @{{.*foo6.*}}(i8 [[A_COERCE_0:%.*]], double [[A_COERCE_1:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[RETVAL:%.*]] = alloca [[UNION_U]], align 8
// CHECK-NEXT:   [[A:%.*]] = alloca [[STRUCT_A]], align 8
// CHECK-NEXT:   [[TMP_0:%.*]] = getelementptr inbounds { i8, double }, ptr [[A]], i32 0, i32 0
// CHECK-NEXT:   store i8 [[A_COERCE_0]], ptr [[TMP_0]], align 8
// CHECK-NEXT:   [[TMP_1:%.*]] = getelementptr inbounds { i8, double }, ptr [[A]], i32 0, i32 1
// CHECK-NEXT:   store double [[A_COERCE_1]], ptr [[TMP_1]], align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[A]], i64 16, i1 false)
// CHECK-NEXT:   [[COERCE_DIVE:%.*]] = getelementptr inbounds [[UNION_U]], ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT:   [[TMP_2:%.*]] = load { i64, double }, ptr [[COERCE_DIVE:%.*]], align 8
// CHECK-NEXT:   ret { i64, double } [[TMP_2]]
U foo6(A a) {
  return U(a);
}

// CHECK: define dso_local void @{{.*foo7.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[D:%.*]] = alloca [[STRUCT_D:%.*]], align 8
// CHECK-NEXT: [[A:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 0
// CHECK-NEXT: [[I]] = getelementptr inbounds [[STRUCT_A:%.*]], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: store i8 1, ptr [[I]], align 8
// CHECK-NEXT: [[J:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// CHECK-NEXT: [[B:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 1
// CHECK-NEXT: [[I1:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: store i8 11, ptr [[I1]], align 8
// CHECK-NEXT: [[J2:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store double 1.100000e+01, ptr [[J2]], align 8
// CHECK-NEXT: [[C:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 3
// CHECK-NEXT: [[I3:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[C]], i32 0, i32 0
// CHECK-NEXT: store i8 111, ptr [[I3]], align 8
// CHECK-NEXT: [[J4:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[C]], i32 0, i32 1
// CHECK-NEXT: store double 1.110000e+02, ptr [[J4]], align 8
// CHECK-NEXT: ret void
void foo7() {
  D d(A(1, 1), A(11, 11), A(111, 111));
}

// CHECK: dso_local void @{{.*foo8.*}}(ptr dead_on_unwind noalias writable sret([[STRUCT_D]]) align 8 [[AGG_RESULT:%.*]])
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[AGG_RESULT]], ptr align 8 [[D1]], i64 56, i1 false)
// CHECK-NEXT: ret void
D foo8() {
  return d1;
}

// CHECK: define dso_local void @{{.*foo9.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[D:%.*]] = alloca [[STRUCT_D:%.*]], align 8
// CHECK-NEXT: [[A:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 0
// CHECK-NEXT: [[I]] = getelementptr inbounds [[STRUCT_A:%.*]], ptr [[A]], i32 0, i32 0
// CHECK-NEXT: store i8 1, ptr [[I]], align 8
// CHECK-NEXT: [[J:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[A]], i32 0, i32 1
// CHECK-NEXT: store double 1.000000e+00, ptr [[J]], align 8
// CHECK-NEXT: [[B:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 1
// CHECK-NEXT: [[I1:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[B]], i32 0, i32 0
// CHECK-NEXT: store i8 2, ptr [[I1]], align 8
// CHECK-NEXT: [[J2:%.*]] = getelementptr inbounds [[STRUCT_A]], ptr [[B]], i32 0, i32 1
// CHECK-NEXT: store double 2.000000e+00, ptr [[J2]], align 8
// CHECK-NEXT: [[C:%.*]] = getelementptr inbounds [[STRUCT_D]], ptr [[D]], i32 0, i32 3
// CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 8 [[C]], i8 0, i64 16, i1 false)
// CHECK-NEXT: ret void
void foo9() {
  D d(A(1, 1));
}

// CHECK: define dso_local noundef ptr @{{.*foo10.*}}()
// CHECK-NEXT: entry:
// CHECK-NEXT: ret ptr [[ARR1]]
const int* foo10() {
  return arr1;
}

// CHECK: define dso_local void @{{.*foo11.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[B_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[ARR_2:%.*]] = alloca [4 x i32], align 16
// CHECK-NEXT: store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// CHECK-NEXT: store i32 [[B:%.*]], ptr [[B_ADDR]], align 4
// CHECK-NEXT: [[TMP_0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_0]], ptr [[ARR_2]], align 4
// CHECK-NEXT: [[ARRINIT_ELEM:%.*]] = getelementptr inbounds i32, ptr [[ARR_2]], i64 1
// CHECK-NEXT: [[TMP_1:%.*]] = load i32, ptr [[B_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_1]], ptr [[ARRINIT_ELEM]], align 4
// CHECK-NEXT: [[ARRINIT_START:%.*]] = getelementptr inbounds i32, ptr [[ARR_2]], i64 2
// CHECK-NEXT: [[ARRINIT_END:%.*]] = getelementptr inbounds i32, ptr [[ARR_2]], i64 4
// CHECK-NEXT: br label [[ARRINIT_BODY:%.*]]
// CHECK: [[ARRINIT_CUR:%.*]] = phi ptr [ [[ARRINIT_START]], %entry ], [ [[ARRINIT_NEXT:%.*]], [[ARRINIT_BODY]] ]
// CHECK-NEXT: store i32 0, ptr [[ARRINIT_CUR]], align 4
// CHECK-NEXT: [[ARRINIT_NEXT]] = getelementptr inbounds i32, ptr [[ARRINIT_CUR]], i64 1
// CHECK-NEXT: [[ARRINIT_DONE:%.*]] = icmp eq ptr [[ARRINIT_NEXT]], [[ARRINIT_END:%.*]]
// CHECK-NEXT: br i1 [[ARRINIT_DONE]], label [[ARRINIT_END1:%.*]], label [[ARRINIT_BODY]]
// CHECK: ret void
void foo11(int a, int b) {
  int arr2[4](a, b);
}

// CHECK: define dso_local void @{{.*foo12.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[A_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[B_ADDR:%.*]] = alloca i32, align 4
// CHECK-NEXT: [[ARR_3:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: store i32 [[A:%.*]], ptr [[A_ADDR]], align 4
// CHECK-NEXT: store i32 [[B:%.*]], ptr [[B_ADDR]], align 4
// CHECK-NEXT: [[TMP_0:%.*]] = load i32, ptr [[A_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_0]], ptr [[ARR_3]], align 4
// CHECK-NEXT: [[ARRINIT_ELEMENT:%.*]] = getelementptr inbounds i32, ptr [[ARR_3]], i64 1
// CHECK-NEXT: [[TMP_1:%.*]] = load i32, ptr [[B_ADDR]], align 4
// CHECK-NEXT: store i32 [[TMP_1]], ptr [[ARRINIT_ELEMENT]], align 4
// CHECK-NEXT: ret void
void foo12(int a, int b) {
  int arr3[](a, b);
}

// CHECK: define dso_local { i8, double } @{{.*foo13.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[RETVAL:%.*]] = alloca [[STRUCT_A]], align 8
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[A2]], i64 16, i1 false)
// CHECK-NEXT: [[TMP_0:%.*]] = load { i8, double }, ptr [[RETVAL]], align 8
// CHECK-NEXT: ret { i8, double } [[TMP_0]]
A foo13() {
  return a2;
}

// CHECK: define dso_local noundef ptr @{{.*foo14.*}}()
// CHECK-NEXT: entry:
// CHECK-NEXT: ret ptr [[ARR4]]
const int* foo14() {
  return arr4;
}

// CHECK: define dso_local noundef ptr @{{.*foo15.*}}()
// CHECK-NEXT: entry:
// CHECK-NEXT: ret ptr [[ARR5]]
const int* foo15() {
  return arr5;
}

// CHECK: define dso_local void @{{.*foo16.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[ARR_6:%.*arr6.*]] = alloca ptr, align 8
// CHECK-NEXT: [[REF_TMP:%.*]] = alloca [1 x i32], align 4
// CHECK-NEXT: store i32 3, ptr [[REF_TMP]], align 4
// CHECK-NEXT: store ptr [[REF_TMP]], ptr [[ARR_6]], align 8
// CHECK-NEXT: ret void
void foo16() {
  int (&&arr6)[] = static_cast<int[]>(3);
}

// CHECK: define dso_local void @{{.*foo17.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[ARR_7:%.*arr7.*]] = alloca ptr, align 8
// CHECK-NEXT: [[REF_TMP:%.*]] = alloca [2 x i32], align 4
// CHECK-NEXT: store i32 4, ptr [[REF_TMP]], align 4
// CHECK-NEXT: [[ARRINIT_START:%.*]] = getelementptr inbounds i32, ptr [[REF_TMP]], i64 1
// CHECK-NEXT: [[ARRINIT_END:%.*]] = getelementptr inbounds i32, ptr [[REF_TMP]], i64 2
// CHECK-NEXT: br label [[ARRINIT_BODY]]
// CHECK: [[ARRINIT_CUR:%.*]] = phi ptr [ [[ARRINIT_START]], %entry ], [ [[ARRINIT_NEXT:%.*]], [[ARRINIT_BODY]] ]
// CHECK-NEXT: store i32 0, ptr [[ARRINIT_CUR]], align 4
// CHECK-NEXT: [[ARRINIT_NEXT]] = getelementptr inbounds i32, ptr [[ARRINIT_CUR]], i64 1
// CHECK-NEXT: [[ARRINIT_DONE:%.*]] = icmp eq ptr [[ARRINIT_NEXT]], [[ARRINIT_END:%.*]]
// CHECK-NEXT: br i1 [[ARRINIT_DONE]], label [[ARRINIT_END1:%.*]], label [[ARRINIT_BODY]]
// CHECK: store ptr [[REF_TMP]], ptr [[ARR_7]], align 8
// CHECK: ret void
void foo17() {
  int (&&arr7)[2] = static_cast<int[2]>(4);
}

// CHECK: define dso_local void @{{.*foo18.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[E:%.*e.*]] = alloca [[STRUCT_E]], align 8
// CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_E]], ptr [[E]], i32 0, i32 0
// CHECK-NEXT: store i32 1, ptr [[A]], align 8
// CHECK-NEXT: [[FN:%.*fn.*]] = getelementptr inbounds [[STRUCT_E]], ptr [[E]], i32 0, i32 1
// CHECK-NEXT: store ptr [[STR]], ptr [[FN]], align 8
// CHECK: ret void
void foo18() {
  E e(1);
}

// CHECK: define dso_local void @{{.*foo19.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[G:%.*g.*]] = alloca [[STRUCT_G]], align 4
// CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_G]], ptr [[G]], i32 0, i32 0
// CHECK-NEXT: store i32 2, ptr [[A]], align 4
// CHECK-NEXT: [[F:%.*f.*]] = getelementptr inbounds [[STRUCT_G]], ptr [[G]], i32 0, i32 1
// CHECk-NEXT: call void @{{.*F.*}}(ptr noundef nonnull align 1 dereferenceable(1)) [[F]], ie32 noundef 1)
// CHECK: ret void
void foo19() {
  G g(2);
}

namespace gh61145 {
  // a.k.a. void make1<0>()
  // CHECK: define {{.*}} void @_ZN7gh611455make1ILi0EEEvv
  // CHECK-NEXT: entry:
  // CHECK-NEXT: [[V:%.*v.*]] = alloca [[STRUCT_VEC]], align 1
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*agg.tmp.ensured.*]] = alloca [[STRUCT_S1]], align 1
  // a.k.a. Vec::Vec()
  // CHECK-NEXT: call void @_ZN7gh611453VecC1Ev(ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // CHECK-NEXT: [[V1:%.*v1.*]] = getelementptr inbounds [[STRUCT_S1]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // a.k.a. Vec::Vec(Vec&&)
  // CHECK-NEXT: call void @_ZN7gh611453VecC1EOS0_(ptr noundef nonnull align 1 dereferenceable(1) [[V1]], ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // a.k.a. S1::~S1()
  // CHECK-NEXT: call void @_ZN7gh611452S1D1Ev(ptr noundef nonnull align 1 dereferenceable(1) [[AGG_TMP_ENSURED]])
  // a.k.a.Vec::~Vec()
  // CHECK-NEXT: call void @_ZN7gh611453VecD1Ev(ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // CHECK-NEXT: ret void
  template <int I>
  void make1() {
    Vec v;
    S1((Vec&&) v);
  }

  // a.k.a. void make1<0>()
  // CHECK: define {{.*}} void @_ZN7gh611455make2ILi0EEEvv
  // CHECK-NEXT: entry:
  // CHECK-NEXT: [[V:%.*v.*]] = alloca [[STRUCT_VEC]], align 1
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*agg.tmp.ensured.*]] = alloca [[STRUCT_S2]], align 1
  // a.k.a. Vec::Vec()
  // CHECK-NEXT: call void @_ZN7gh611453VecC1Ev(ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // CHECK-NEXT: [[V1:%.*v1.*]] = getelementptr inbounds [[STRUCT_S2]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // a.k.a. Vec::Vec(Vec&&)
  // CHECK-NEXT: call void @_ZN7gh611453VecC1EOS0_(ptr noundef nonnull align 1 dereferenceable(1) [[V1]], ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // CHECK-NEXT: [[C:%.*c.*]] = getelementptr inbounds [[STRUCT_S2]], ptr [[AGG_TMP_ENSURED]], i32 0, i32
  // CHECK-NEXT: store i8 0, ptr [[C]], align 1
  // a.k.a. S2::~S2()
  // CHECK-NEXT: call void @_ZN7gh611452S2D1Ev(ptr noundef nonnull align 1 dereferenceable(2) [[AGG_TMP_ENSURED]])
  // a.k.a. Vec::~Vec()
  // CHECK-NEXT: call void @_ZN7gh611453VecD1Ev(ptr noundef nonnull align 1 dereferenceable(1) [[V]])
  // CHECK-NEXT: ret void
  template <int I>
  void make2() {
    Vec v;
    S2((Vec&&) v, 0);
  }

  void foo() {
    make1<0>();
    make2<0>();
  }
}

namespace gh62266 {
  // CHECK: define {{.*}} void {{.*foo20.*}}
  // CHECK-NEXT: entry:
  // CHECK-NEXT: [[H:%.*h.*]] = alloca [[STRUCT_H]], align 4
  // CHECK-NEXT: [[I:%.*i.*]] = getelementptr inbounds [[STRUCT_H]], ptr [[H]], i32 0, i32 0
  // CHECK-NEXT: store i32 1, ptr [[I]], align 4
  // CHECK-NEXT: [[J:%.*j.*]] = getelementptr inbounds [[STRUCT_H]], ptr [[H]], i32 0, i32 1
  // CHECK-NEXT: store i32 2, ptr [[J]], align 4
  // CHECK-NEXT: ret void
  void foo20() {
    H<2> h(1);
  }
}

namespace gh61567 {
  int foo20();

  // CHECK: define {{.*}} void @{{.*foo21.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]], align 8
  // CHECK-NEXT: [[REF_TMP:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // CHECK-NEXT: store i32 0, ptr [[A]], align 8
  // CHECK-NEXT: [[R:%.*r.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // CHECK-NEXT: store i32 1, ptr [[REF_TMP]], align 4
  // CHECK-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // CHECK-NEXT: ret void
  void foo21() {
    I(0, 1);
  }

  // CHECK: define {{.*}} void @{{.*foo22.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]], align 8
  // CHECK-NEXT: [[REF_TMP:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // CHECK-NEXT: store i32 0, ptr [[A]], align 8
  // CHECK-NEXT: [[R:%.*r.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // CHECK-NEXT: [[CALL:%.*call*]] = call noundef i32 @{{.*foo20.*}}
  // CHECK-NEXT: store i32 [[CALL]], ptr [[REF_TMP]], align 4
  // CHECK-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // CHECK-NEXT: ret void
  void foo22() {
    I(0, foo20());
  }

  // CHECK: define {{.*}} void @{{.*foo23.*}}(i32 noundef [[I:%.*i.*]])
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[I_ADDR:%.*i.*]] = alloca i32, align 4
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]], align 8
  // CHECK-NEXT: store i32 [[I]], ptr [[I_ADDR]], align 4
  // CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // CHECK-NEXT: store i32 0, ptr [[A]], align 8
  // CHECK-NEXT: [[R:%.*r.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // CHECK-NEXT: store ptr [[I_ADDR]], ptr [[R]], align 8
  // CHECK-NEXT: ret void
  void foo23(int i) {
    I(0, static_cast<int&&>(i));
  }

  // CHECK: define {{.*}} void @{{.*foo24.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[AGG_TMP_ENSURED:%.*]] = alloca [[STRUCT_I]], align 8
  // CHECK-NEXT: [[REF_TMP:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[A:%.*a.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 0
  // CHECK-NEXT: store i32 0, ptr [[A]], align 8
  // CHECK-NEXT: [[R:%.*r.*]] = getelementptr inbounds [[STRUCT_I]], ptr [[AGG_TMP_ENSURED]], i32 0, i32 1
  // CHECK-NEXT: store i32 2, ptr [[REF_TMP]], align 4
  // CHECK-NEXT: store ptr [[REF_TMP]], ptr [[R]], align 8
  // CHECK-NEXT: ret void
  void foo24() {
    I(0);
  }
}

namespace gh68198 {
  // CHECK: define {{.*}} void @{{.*foo25.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[ARR_8:%.*arr8.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[CALL_PTR:%.*]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 8)
  // CHECK-NEXT: store i32 1, ptr [[CALL_PTR]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr inbounds i32, ptr [[CALL_PTR]], i64 1
  // CHECK-NEXT: store i32 2, ptr [[ARRAY_EXP_NEXT]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT1:%.*]] = getelementptr inbounds i32, ptr [[ARRAY_EXP_NEXT]], i64 1
  // CHECK-NEXT: store ptr [[CALL_PTR]], ptr %arr8, align 8
  // CHECK-NEXT: ret void
  void foo25() {
    int* arr8 = new int[](1, 2);
  }

  // CHECK: define {{.*}} void @{{.*foo26.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[ARR_10:%.*arr9.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[CALL_PTR]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 16)
  // CHECK-NEXT: store i32 1, ptr [[CALL]], align 4
  // CHECK-NEXT: [[ARRAYINIT_ELEMENT:%.*]] = getelementptr inbounds i32, ptr [[CALL]], i64 1
  // CHECK-NEXT: store i32 2, ptr [[ARRAYINIT_ELEMENT]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr inbounds [2 x i32], ptr %call, i64 1
  // CHECK-NEXT: store i32 3, ptr [[ARRAY_EXP_NEXT]], align 4
  // CHECK-NEXT: [[ARRAYINIT_ELEMENT2:%.*]] = getelementptr inbounds i32, ptr [[ARRAY_EXP_NEXT]], i64 1
  // CHECK-NEXT: store i32 4, ptr [[ARRAYINIT_ELEMENT2]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT3:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARRAY_EXP_NEXT]], i64 1
  // CHECK-NEXT: store ptr [[CALL_PTR]], ptr [[ARR_10]], align 8
  // CHECK-NEXT: ret void
  void foo26() {
    void* arr9 = new int[][2]({1, 2}, {3, 4});
  }

  // CHECK: define {{.*}} void @{{.*foo27.*}} {
  // CHECK-NEXT: entry
  // CHECK-NEXT: [[ARR_10:%.*arr10.*]] = alloca ptr, align 8
  // CHECK-NEXT: [[CALL_PTR]] = call noalias noundef nonnull ptr @_Znam(i64 noundef 32)
  // CHECK-NEXT: store i32 5, ptr [[CALL]], align 4
  // CHECK-NEXT: [[ARRAYINIT_ELEMENT:%.*]] = getelementptr inbounds i32, ptr [[CALL]], i64 1
  // CHECK-NEXT: store i32 6, ptr [[ARRAYINIT_ELEMENT]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT:%.*]] = getelementptr inbounds [2 x i32], ptr %call, i64 1
  // CHECK-NEXT: store i32 7, ptr [[ARRAY_EXP_NEXT]], align 4
  // CHECK-NEXT: [[ARRAYINIT_ELEMENT2:%.*]] = getelementptr inbounds i32, ptr [[ARRAY_EXP_NEXT]], i64 1
  // CHECK-NEXT: store i32 8, ptr [[ARRAYINIT_ELEMENT2]], align 4
  // CHECK-NEXT: [[ARRAY_EXP_NEXT3:%.*]] = getelementptr inbounds [2 x i32], ptr [[ARRAY_EXP_NEXT]], i64 1
  // CHECK-NEXT: call void @llvm.memset.p0.i64(ptr align 4 [[ARRAY_EXP_NEXT3]], i8 0, i64 16, i1 false)
  // CHECK-NEXT: store ptr [[CALL_PTR]], ptr [[ARR_10]], align 8
  // CHECK-NEXT: ret void
  void foo27() {
    void* arr10 = new int[4][2]({5, 6}, {7, 8});
  }
}
