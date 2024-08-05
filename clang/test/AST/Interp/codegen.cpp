// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s


int arr[2];
// CHECK: @pastEnd = constant ptr getelementptr (i8, ptr @arr, i64 8)
int &pastEnd = arr[2];

// CHECK: @F = constant ptr @arr, align 8
int &F = arr[0];

struct S {
  int a;
  float c[3];
};

// CHECK: @s = global %struct.S zeroinitializer, align 4
S s;
// CHECK: @sp = constant ptr getelementptr (i8, ptr @s, i64 16), align 8
float &sp = s.c[3];


namespace BaseClassOffsets {
  struct A { int a; };
  struct B { int b; };
  struct C : A, B { int c; };

  extern C c;
  // CHECK: @_ZN16BaseClassOffsets1aE = global ptr @_ZN16BaseClassOffsets1cE, align 8
  A* a = &c;
  // CHECK: @_ZN16BaseClassOffsets1bE = global ptr getelementptr (i8, ptr @_ZN16BaseClassOffsets1cE, i64 4), align 8
  B* b = &c;
}

namespace ExprBase {
  struct A { int n; };
  struct B { int n; };
  struct C : A, B {};

  extern const int &&t = ((B&&)C{}).n;
  // CHECK: @_ZGRN8ExprBase1tE_ = internal global {{.*}} zeroinitializer,
  // CHECK: @_ZN8ExprBase1tE = constant ptr {{.*}} @_ZGRN8ExprBase1tE_, {{.*}} 8
}

namespace reinterpretcast {
  const unsigned int n = 1234;
  extern const int &s = reinterpret_cast<const int&>(n);
  // CHECK: @_ZN15reinterpretcastL1nE = internal constant i32 1234, align 4
  // CHECK: @_ZN15reinterpretcast1sE = constant ptr @_ZN15reinterpretcastL1nE, align 8

  void *f1(unsigned long l) {
    return reinterpret_cast<void *>(l);
  }
  // CHECK: define {{.*}} ptr @_ZN15reinterpretcast2f1Em
  // CHECK: inttoptr
}

namespace Bitfield {
  struct S { int a : 5; ~S(); };
  // CHECK: alloca
  // CHECK: call {{.*}}memset
  // CHECK: store i32 {{.*}}, ptr @_ZGRN8Bitfield1rE_
  // CHECK: call void @_ZN8Bitfield1SD1
  // CHECK: store ptr @_ZGRN8Bitfield1rE_, ptr @_ZN8Bitfield1rE, align 8
  int &&r = S().a;
}
