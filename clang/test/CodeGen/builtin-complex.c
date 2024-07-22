// RUN: %clang_cc1 -triple x86_64-linux -w -o - -emit-llvm -DT=float %s | FileCheck %s --check-prefixes=CHECK,CHECK-FLOAT
// RUN: %clang_cc1 -triple x86_64-linux -w -o - -emit-llvm -DT=double %s | FileCheck %s --check-prefixes=CHECK,CHECK-DOUBLE
// RUN: %clang_cc1 -triple x86_64-linux -w -o - -emit-llvm -DT="long double" %s | FileCheck %s --check-prefixes=CHECK,CHECK-FP80
// RUN: %clang_cc1 -triple x86_64-linux -w -o - -emit-llvm -DT=__float128 %s | FileCheck %s --check-prefixes=CHECK,CHECK-FP128
// FIXME: If we start to support _Complex __fp16 or _Complex _Float16, add tests for them too.

// CHECK-FLOAT: @global ={{.*}} global { [[T:float]], [[T]] } { [[T]] 1.0{{.*}}, [[T]] 2.0{{.*}} }
// CHECK-DOUBLE: @global ={{.*}} global { [[T:double]], [[T]] } { [[T]] 1.0{{.*}}, [[T]] 2.0{{.*}} }
// CHECK-FP80: @global ={{.*}} global { [[T:x86_fp80]], [[T]] } { [[T]] 0xK3FFF8000000000000000, [[T]] 0xK40008000000000000000 }
// CHECK-FP128: @global ={{.*}} global { [[T:fp128]], [[T]] } { [[T]] 0xL00000000000000003FFF000000000000, [[T]] 0xL00000000000000004000000000000000 }
_Complex T global = __builtin_complex(1.0, 2.0);

// CHECK-LABEL: @test
_Complex T test(T a, T b) {
  return __builtin_complex(a, b);
  // CHECK: %[[A:.*]] = load [[T]], ptr %a.addr,
  // CHECK: %[[B:.*]] = load [[T]], ptr %b.addr,
  // CHECK: %[[RET_RE:.*]] = getelementptr inbounds { [[T]], [[T]] }, ptr %[[RET:[^,]*]], i32 0, i32 0
  // CHECK: %[[RET_IM:.*]] = getelementptr inbounds { [[T]], [[T]] }, ptr %[[RET]], i32 0, i32 1
  // CHECK: store [[T]] %[[A]], ptr %[[RET_RE]],
  // CHECK: store [[T]] %[[B]], ptr %[[RET_IM]],
}
