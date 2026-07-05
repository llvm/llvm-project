// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm -target-feature +altivec -target-cpu pwr7 -o - %s | FileCheck %s --check-prefixes=CHECK,AIX32
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm -target-feature +altivec -target-cpu pwr7 -o - %s | FileCheck %s --check-prefixes=CHECK,AIX64

vector double vector_varargs(int count, ...) {
  __builtin_va_list arg_list;
  __builtin_va_start(arg_list, count);

  vector double ret;

  for (int i = 0; i != count; ++i) {
    ret = __builtin_va_arg(arg_list, vector double);
  }

  __builtin_va_end(arg_list);
  return ret;
}

// CHECK:         %arg_list = alloca ptr
// CHECK:         call void @llvm.va_start.p0(ptr %arg_list)

// AIX32:       for.body:
// AIX32-NEXT:    %argp.cur = load ptr, ptr %arg_list, align 4
// AIX32-NEXT:    %2 = getelementptr inbounds i8, ptr %argp.cur, i32 15
// AIX32-NEXT:    %argp.cur.aligned = call ptr @llvm.ptrmask.p0.i32(ptr %2, i32 -16)
// AIX32-NEXT:    %argp.next = getelementptr inbounds i8, ptr %argp.cur.aligned, i32 16
// AIX32-NEXT:    store ptr %argp.next, ptr %arg_list, align 4
// AIX32-NEXT:    %3 = load <2 x double>, ptr %argp.cur.aligned, align 16
// AIX32-NEXT:    store <2 x double> %3, ptr %ret, align 16
// AIX32-NEXT:    br label %for.inc

// AIX64:       for.body:
// AIX64-NEXT:    %argp.cur = load ptr, ptr %arg_list, align 8
// AIX64-NEXT:    %2 = getelementptr inbounds i8, ptr %argp.cur, i32 15
// AIX64-NEXT:    %argp.cur.aligned = call ptr @llvm.ptrmask.p0.i64(ptr %2, i64 -16)
// AIX64-NEXT:    %argp.next = getelementptr inbounds i8, ptr %argp.cur.aligned, i64 16
// AIX64-NEXT:    store ptr %argp.next, ptr %arg_list, align 8
// AIX64-NEXT:    %3 = load <2 x double>, ptr %argp.cur.aligned, align 16
// AIX64-NEXT:    store <2 x double> %3, ptr %ret, align 16
// AIX64-NEXT:    br label %for.inc


// CHECK:      for.end:
// CHECK:        call void @llvm.va_end.p0(ptr %arg_list)
