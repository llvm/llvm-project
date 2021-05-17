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

// CHECK:         %arg_list = alloca i8*
// CHECK:         %arg_list1 = bitcast i8** %arg_list to i8*
// CHECK:         call void @llvm.va_start(i8* %arg_list1)

// AIX32:       for.body:
// AIX32-NEXT:    %argp.cur = load i8*, i8** %arg_list, align 4
// AIX32-NEXT:    %2 = ptrtoint i8* %argp.cur to i32
// AIX32-NEXT:    %3 = add i32 %2, 15
// AIX32-NEXT:    %4 = and i32 %3, -16
// AIX32-NEXT:    %argp.cur.aligned = inttoptr i32 %4 to i8*
// AIX32-NEXT:    %argp.next = getelementptr inbounds i8, i8* %argp.cur.aligned, i32 16
// AIX32-NEXT:    store i8* %argp.next, i8** %arg_list, align 4
// AIX32-NEXT:    %5 = bitcast i8* %argp.cur.aligned to <2 x double>*
// AIX32-NEXT:    %6 = load <2 x double>, <2 x double>* %5, align 16
// AIX32-NEXT:    store <2 x double> %6, <2 x double>* %ret, align 16
// AIX32-NEXT:    br label %for.inc

// AIX64:       for.body:
// AIX64-NEXT:    %argp.cur = load i8*, i8** %arg_list, align 8
// AIX64-NEXT:    %2 = ptrtoint i8* %argp.cur to i64
// AIX64-NEXT:    %3 = add i64 %2, 15
// AIX64-NEXT:    %4 = and i64 %3, -16
// AIX64-NEXT:    %argp.cur.aligned = inttoptr i64 %4 to i8*
// AIX64-NEXT:    %argp.next = getelementptr inbounds i8, i8* %argp.cur.aligned, i64 16
// AIX64-NEXT:    store i8* %argp.next, i8** %arg_list, align 8
// AIX64-NEXT:    %5 = bitcast i8* %argp.cur.aligned to <2 x double>*
// AIX64-NEXT:    %6 = load <2 x double>, <2 x double>* %5, align 16
// AIX64-NEXT:    store <2 x double> %6, <2 x double>* %ret, align 16
// AIX64-NEXT:    br label %for.inc


// CHECK:      for.end:
// CHECK:        %arg_list2 = bitcast i8** %arg_list to i8*
// CHECK:        call void @llvm.va_end(i8* %arg_list2)
