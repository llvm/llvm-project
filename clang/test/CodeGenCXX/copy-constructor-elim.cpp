// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %ms_abi_triple -emit-llvm -o - %s | FileCheck %s -check-prefix MS
// CHECK-NOT: _ZN1CC1ERK1C
// CHECK-NOT: _ZN1SC1ERK1S
// MS-NOT: ?0C@@QAE@ABV0
// MS-NOT: ?0S@@QAE@ABV0

extern "C" int printf(...);


struct C {
  C() : iC(6) {printf("C()\n"); }
  C(const C& c) { printf("C(const C& c)\n"); }
  int iC;
};

C foo() {
  return C();
};

class X { // ...
public: 
  X(int) {}
  X(const X&, int i = 1, int j = 2, C c = foo()) {
    printf("X(const X&, %d, %d, %d)\n", i, j, c.iC);
  }
};


struct S {
  S();
};

S::S() { printf("S()\n"); }

void Call(S) {};

int main() {
  X a(1);
  X b(a, 2);
  X c = b;
  X d(a, 5, 6);
  S s;
  Call(s);
}

struct V {
  int x;
};

typedef V V_over_aligned __attribute((aligned(8)));
extern const V_over_aligned gv1 = {};

extern "C" V f() { return gv1; }

// Make sure that we obey the destination's alignment requirements when emitting
// the copy.
// CHECK-LABEL: define {{.*}} @f(
// CHECK:   call void @llvm.memcpy.p0.p0.{{i64|i32}}({{.*}}align 4{{.*}}, ptr align 8 @gv1, {{i64|i32}} 4, i1 false)
