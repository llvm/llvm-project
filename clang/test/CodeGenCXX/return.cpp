// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -o - %s | FileCheck --check-prefixes=CHECK,CHECK-COMMON %s
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -O -o - %s | FileCheck %s --check-prefixes=CHECK-OPT,CHECK-COMMON
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -fno-strict-return -o - %s | FileCheck %s --check-prefixes=CHECK-NOSTRICT,CHECK-COMMON
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -fno-strict-return -Wno-return-type -o - %s | FileCheck %s --check-prefixes=CHECK-NOSTRICT,CHECK-COMMON
// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -std=c++11 -fno-strict-return -O -o - %s | FileCheck %s --check-prefixes=CHECK-NOSTRICT-OPT,CHECK-COMMON

// CHECK-COMMON-LABEL: @_Z9no_return
int no_return() {
  // CHECK:      call void @llvm.trap
  // CHECK-NEXT: unreachable

  // CHECK-OPT-NOT: call void @llvm.trap
  // CHECK-OPT:     unreachable

  // -fno-strict-return should not emit trap + unreachable but it should return
  // an undefined value instead.

  // CHECK-NOSTRICT: alloca
  // CHECK-NOSTRICT-NEXT: load
  // CHECK-NOSTRICT-NEXT: ret i32
  // CHECK-NOSTRICT-NEXT: }

  // CHECK-NOSTRICT-OPT: ret i32 undef
}

enum Enum {
  A, B
};

// CHECK-COMMON-LABEL: @_Z27returnNotViableDontOptimize4Enum
int returnNotViableDontOptimize(Enum e) {
  switch (e) {
  case A: return 1;
  case B: return 2;
  }
  // Undefined behaviour optimization shouldn't be used when -fno-strict-return
  // is turned on, even if all the enum cases are covered in this function.

  // CHECK-NOSTRICT-NOT: call void @llvm.trap
  // CHECK-NOSTRICT-NOT: unreachable
}

struct Trivial {
  int x;
};

// CHECK-NOSTRICT-LABEL: @_Z7trivialv
Trivial trivial() {
  // This function returns a trivial record so -fno-strict-return should avoid
  // the undefined behaviour optimization.

  // CHECK-NOSTRICT-NOT: call void @llvm.trap
  // CHECK-NOSTRICT-NOT: unreachable
}

struct NonTrivialCopy {
  NonTrivialCopy(const NonTrivialCopy &);
};

// CHECK-NOSTRICT-LABEL: @_Z14nonTrivialCopyv
NonTrivialCopy nonTrivialCopy() {
  // CHECK-NOSTRICT-NOT: call void @llvm.trap
  // CHECK-NOSTRICT-NOT: unreachable
}

struct NonTrivialDefaultConstructor {
  int x;

  NonTrivialDefaultConstructor() { }
};

// CHECK-NOSTRICT-LABEL: @_Z28nonTrivialDefaultConstructorv
NonTrivialDefaultConstructor nonTrivialDefaultConstructor() {
  // CHECK-NOSTRICT-NOT: call void @llvm.trap
  // CHECK-NOSTRICT-NOT: unreachable
}

// Functions that return records with non-trivial destructors should always use
// the -fstrict-return optimization.

struct NonTrivialDestructor {
  ~NonTrivialDestructor();
};

// CHECK-NOSTRICT-LABEL: @_Z20nonTrivialDestructorv
NonTrivialDestructor nonTrivialDestructor() {
  // CHECK-NOSTRICT: call void @llvm.trap
  // CHECK-NOSTRICT-NEXT: unreachable
}

// The behavior for lambdas should be identical to functions.
// CHECK-COMMON-LABEL: @_Z10lambdaTestv
void lambdaTest() {
  auto lambda1 = []() -> int {
  };
  lambda1();

  // CHECK: call void @llvm.trap
  // CHECK-NEXT: unreachable

  // CHECK-NOSTRICT-NOT: call void @llvm.trap
  // CHECK-NOSTRICT-NOT: unreachable
}
