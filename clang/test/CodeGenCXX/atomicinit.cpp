// RUN: %clang_cc1 -fno-inline-functions %s -emit-llvm -O1 -o - -triple=i686-apple-darwin9 -std=c++11 | FileCheck %s

// CHECK-DAG: @PR22043 ={{.*}} local_unnamed_addr global i32 0, align 4
typedef _Atomic(int) AtomicInt;
AtomicInt PR22043 = AtomicInt();

// CHECK-DAG: @_ZN7PR180978constant1aE ={{.*}} local_unnamed_addr global { i16, i8 } { i16 1, i8 6 }, align 4
// CHECK-DAG: @_ZN7PR180978constant1bE ={{.*}} local_unnamed_addr global { i16, i8 } { i16 2, i8 6 }, align 4
// CHECK-DAG: @_ZN7PR180978constant1cE ={{.*}} local_unnamed_addr global { i16, i8 } { i16 3, i8 6 }, align 4
// CHECK-DAG: @_ZN7PR180978constant1yE ={{.*}} local_unnamed_addr global { { i16, i8 }, i32 } { { i16, i8 } { i16 4, i8 6 }, i32 5 }, align 4
// CHECK-DAG: @_ZN7PR180978constant1zE ={{.*}} global i32 0, align 4
// CHECK-DAG: @_ZN7PR180978constant2y2E ={{.*}} global %"struct.PR18097::constant::Y" zeroinitializer
// CHECK-DAG: @_ZN7PR180978constant1gE ={{.*}} global %"struct.PR18097::constant::Struct0" zeroinitializer
// CHECK-DAG: @_ZN7PR180978constantL1xE = internal {{.*}} constant { i16, i8 } { i16 1, i8 6 }

struct A {
  _Atomic(int) i;
  A(int j);
  void v(int j);
};
// Storing to atomic values should be atomic
// CHECK: store atomic i32 {{.*}} seq_cst, align 4
void A::v(int j) { i = j; }
// Initialising atomic values should not be atomic
// CHECK-NOT: store atomic 
A::A(int j) : i(j) {}

struct B {
  int i;
  B(int x) : i(x) {}
};

_Atomic(B) b;

// CHECK-LABEL: define{{.*}} void @_Z11atomic_initR1Ai
void atomic_init(A& a, int i) {
  // CHECK-NOT: atomic
  // CHECK: call void @_ZN1BC1Ei
  __c11_atomic_init(&b, B(i));
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z16atomic_init_boolPU7_Atomicbb
void atomic_init_bool(_Atomic(bool) *ab, bool b) {
  // CHECK-NOT: atomic
  // CHECK: {{zext i1.*to i8}}
  // CHECK-NEXT: store i8
  __c11_atomic_init(ab, b);
  // CHECK-NEXT: ret void
}

struct AtomicBoolMember {
  _Atomic(bool) ab;
  AtomicBoolMember(bool b);
};

// CHECK-LABEL: define{{.*}} void @_ZN16AtomicBoolMemberC2Eb
// CHECK: zext i1 {{.*}} to i8
// CHECK: store i8
// CHECK-NEXT: ret void
AtomicBoolMember::AtomicBoolMember(bool b) : ab(b) { }

namespace PR18097 {
  namespace dynamic {
    struct X {
      X(int);
      short n;
      char c;
    };

    // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
    // CHECK: call void @_ZN7PR180977dynamic1XC1Ei(ptr noundef {{[^,]*}} @_ZN7PR180977dynamic1aE, i32 noundef 1)
    _Atomic(X) a = X(1);

    // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
    // CHECK: call void @_ZN7PR180977dynamic1XC1Ei(ptr noundef {{[^,]*}} @_ZN7PR180977dynamic1bE, i32 noundef 2)
    _Atomic(X) b(X(2));

    // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
    // CHECK: call void @_ZN7PR180977dynamic1XC1Ei(ptr noundef {{[^,]*}} @_ZN7PR180977dynamic1cE, i32 noundef 3)
    _Atomic(X) c{X(3)};

    struct Y {
      _Atomic(X) a;
      _Atomic(int) b;
    };
    // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
    // CHECK: call void @_ZN7PR180977dynamic1XC1Ei(ptr {{[^,]*}} @_ZN7PR180977dynamic1yE, i32 noundef 4)
    // CHECK: store i32 5, ptr getelementptr inbounds ({{.*}}, ptr @_ZN7PR180977dynamic1yE, i32 0, i32 1)
    Y y = { X(4), 5 };
  }

  // CHECKs at top of file.
  namespace constant {
    struct X {
      constexpr X(int n) : n(n) {}
      short n;
      char c = 6;
    };
    _Atomic(X) a = X(1);
    _Atomic(X) b(X(2));
    _Atomic(X) c{X(3)};

    struct Y {
      _Atomic(X) a;
      _Atomic(int) b;
    };
    Y y = { X(4), 5 };

    // CHECK-LABEL: define {{.*}} @__cxx_global_var_init
    // CHECK: tail call void @llvm.memcpy.p0.p0.i32(ptr{{.*}} @_ZN7PR180978constant2y2E, ptr{{.*}} @_ZN7PR180978constantL1xE, i32 3, i1 false)
    // CHECK: %0 = load i32, ptr @_ZN7PR180978constant1zE
    // CHECK: store i32 %0, ptr getelementptr inbounds (%"struct.PR18097::constant::Y", ptr @_ZN7PR180978constant2y2E, i32 0, i32 1)
    int z;
    constexpr X x{1};
    Y y2 = { x, z };

    typedef union {
      unsigned int f0;
    } Union0;

    typedef struct {
      _Atomic(Union0) f1;
    } Struct0;

    Struct0 g = {};
  }
}
