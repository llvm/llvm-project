// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// rdar://8818236
namespace rdar8818236 {
struct S {
  char c2;
  union {
    char c;
    int i;
  };
};

// CHECK: @_ZN11rdar88182363fooE ={{.*}} global i64 4
char S::*foo  = &S::c;
}

struct A {
  union {
    int a;
    void* b;
  };
  
  A() : a(0) { }
};

A a;

namespace PR7021 {
  struct X
  {
    union { long l; };
  };

  // CHECK-LABEL: define{{.*}} void @_ZN6PR70211fENS_1XES0_
  void f(X x, X z) {
    X x1;

    // CHECK: store i64 1, i64
    x1.l = 1;

    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    X x2(x1);

    X x3;
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    x3 = x1;

    // CHECK: ret void
  }
}

namespace test2 {
  struct A {
    struct {
      union {
        int b;
      };
    };

    A();
  };

  A::A() : b(10) { }
  // CHECK-LABEL: define{{.*}} void @_ZN5test21AC2Ev(
  // CHECK-NOT: }
  // CHECK: store i32 10
  // CHECK: }
}

namespace PR10512 {
  struct A {
    A();
    A(int);
    A(long);

    struct {
      struct {int x;};
      struct {int y;};
    };
  };

  // CHECK-LABEL: define{{.*}} void @_ZN7PR105121AC2Ev
  // CHECK: [[THISADDR:%[a-zA-Z0-9.]+]] = alloca [[A:%"struct[A-Za-z0-9:.]+"]]
  // CHECK-NEXT: store [[A]]* [[THIS:%[a-zA-Z0-9.]+]], [[A]]** [[THISADDR]]
  // CHECK-NEXT: [[THIS1:%[a-zA-Z0-9.]+]] = load [[A]]*, [[A]]** [[THISADDR]]
  // CHECK-NEXT: ret void
  A::A() {}

  // CHECK-LABEL: define{{.*}} void @_ZN7PR105121AC2Ei
  // CHECK: [[THISADDR:%[a-zA-Z0-9.]+]] = alloca [[A:%"struct[A-Za-z0-9:.]+"]]
  // CHECK-NEXT: [[XADDR:%[a-zA-Z0-9.]+]] = alloca i32
  // CHECK-NEXT: store [[A]]* [[THIS:%[a-zA-Z0-9.]+]], [[A]]** [[THISADDR]]
  // CHECK-NEXT: store i32 [[X:%[a-zA-Z0-9.]+]], i32* [[XADDR]]
  // CHECK-NEXT: [[THIS1:%[a-zA-Z0-9.]+]] = load [[A]]*, [[A]]** [[THISADDR]]
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i32, i32* [[XADDR]]
  // CHECK-NEXT: store i32 [[TMP]]
  // CHECK-NEXT: ret void
  A::A(int x) : x(x) { }

  // CHECK-LABEL: define{{.*}} void @_ZN7PR105121AC2El
  // CHECK: [[THISADDR:%[a-zA-Z0-9.]+]] = alloca [[A:%"struct[A-Za-z0-9:.]+"]]
  // CHECK-NEXT: [[XADDR:%[a-zA-Z0-9.]+]] = alloca i64
  // CHECK-NEXT: store [[A]]* [[THIS:%[a-zA-Z0-9.]+]], [[A]]** [[THISADDR]]
  // CHECK-NEXT: store i64 [[X:%[a-zA-Z0-9.]+]], i64* [[XADDR]]
  // CHECK-NEXT: [[THIS1:%[a-zA-Z0-9.]+]] = load [[A]]*, [[A]]** [[THISADDR]]
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 1}}
  // CHECK-NEXT: {{getelementptr inbounds.*i32 0, i32 0}}
  // CHECK-NEXT: [[TMP:%[a-zA-Z0-9.]+]] = load i64, i64* [[XADDR]]
  // CHECK-NEXT: [[CONV:%[a-zA-Z0-9.]+]] = trunc i64 [[TMP]] to i32
  // CHECK-NEXT: store i32 [[CONV]]
  // CHECK-NEXT: ret void
  A::A(long y) : y(y) { }
}

namespace test3 {
  struct A {
    union {
      mutable char fibers[100];
      struct {
        void (*callback)(void*);
        void *callback_value;
      };
    };

    A();
  };

  A::A() : callback(0), callback_value(0) {}
  // CHECK-LABEL: define{{.*}} void @_ZN5test31AC2Ev(
  // CHECK: [[THIS:%.*]] = load
  // CHECK-NEXT: [[UNION:%.*]] = getelementptr inbounds {{.*}} [[THIS]], i32 0, i32 0
  // CHECK-NEXT: [[STRUCT:%.*]] = bitcast {{.*}}* [[UNION]] to 
  // CHECK-NEXT: [[CALLBACK:%.*]] = getelementptr inbounds {{.*}} [[STRUCT]], i32 0, i32 0
  // CHECK: store 
  // CHECK-NEXT: [[UNION:%.*]] = getelementptr inbounds {{.*}} [[THIS]], i32 0, i32 0
  // CHECK-NEXT: [[STRUCT:%.*]] = bitcast {{.*}}* [[UNION]] to 
  // CHECK-NEXT: [[CVALUE:%.*]] = getelementptr inbounds {{.*}} [[STRUCT]], i32 0, i32 1
  // CHECK-NEXT: store i8* null, i8** [[CVALUE]]
}

struct S {
  // CHECK: store i32 42
  // CHECK: store i32 55
  S() : x(42), y(55) {}
  union {
    struct {
      int x;
      union { int y; };
    };
  };
} s;


//PR8760 
template <typename T> struct Foo {
  Foo() : ptr(__nullptr) {}
  union {
    T *ptr;
  };
};
Foo<int> f;

namespace PR9683 {
  struct QueueEntry {
    union {
      struct {
        void* mPtr;
        union {
          unsigned mSubmissionTag;
        };
      };
      unsigned mValue;
    };
    QueueEntry() {}
  };
  QueueEntry QE;
}

namespace PR13154 {
  struct IndirectReferenceField {
      struct {
          float &x;
      };
      IndirectReferenceField(float &x);
  };
  IndirectReferenceField::IndirectReferenceField(float &xx) : x(xx) {}
}
