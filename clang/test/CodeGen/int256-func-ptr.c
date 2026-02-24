// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=AARCH64

// Verify __int256 works correctly through function pointers and extern decls.

typedef __int256 (*binop_t)(__int256, __int256);
typedef int (*pred_t)(__int256, __int256);

// X86-LABEL: define{{.*}} void @call_binop(ptr{{.*}}sret(i256){{.*}}, ptr noundef %fn, ptr noundef byval(i256) align 16 %0, ptr noundef byval(i256) align 16 %1)
// X86: call void %{{.*}}(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %{{.*}})
// AARCH64-LABEL: define{{.*}} i256 @call_binop(ptr noundef %fn, i256 noundef %a, i256 noundef %b)
// AARCH64: call i256 %{{.*}}(i256 noundef %{{.*}}, i256 noundef %{{.*}})
__int256 call_binop(binop_t fn, __int256 a, __int256 b) {
  return fn(a, b);
}

// X86-LABEL: define{{.*}} i32 @call_pred(ptr noundef %fn, ptr noundef byval(i256) align 16 %0, ptr noundef byval(i256) align 16 %1)
// X86: call i32 %{{.*}}(ptr noundef byval(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %{{.*}})
// AARCH64-LABEL: define{{.*}} i32 @call_pred(ptr noundef %fn, i256 noundef %a, i256 noundef %b)
// AARCH64: call i32 %{{.*}}(i256 noundef %{{.*}}, i256 noundef %{{.*}})
int call_pred(pred_t fn, __int256 a, __int256 b) {
  return fn(a, b);
}

// Cross-TU: extern function with __int256 params
extern __int256 extern_add(__int256 a, __int256 b);

// X86-LABEL: define{{.*}} void @call_extern(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %0, ptr noundef byval(i256) align 16 %1)
// X86: call void @extern_add(ptr{{.*}}sret(i256){{.*}}, ptr noundef byval(i256) align 16 %{{.*}}, ptr noundef byval(i256) align 16 %{{.*}})
// AARCH64-LABEL: define{{.*}} i256 @call_extern(i256 noundef %a, i256 noundef %b)
// AARCH64: call i256 @extern_add(i256 noundef %{{.*}}, i256 noundef %{{.*}})
__int256 call_extern(__int256 a, __int256 b) {
  return extern_add(a, b);
}
