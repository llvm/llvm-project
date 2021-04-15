// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm < %s | FileCheck -check-prefixes=LINUX,COMMON %s
// RUN: %clang_cc1 -triple aarch64-pc-win32 -emit-llvm < %s | FileCheck -check-prefixes=WIN64,COMMON %s

struct small_odd {
  char a, b, c;
};

struct larger {
  int a, b, c, d, e;
};

void __attribute__((ms_abi)) f1(void);
void f2(void);
void f3(void) {
  // LINUX-LABEL: define{{.*}} void @f3()
  // WIN64-LABEL: define dso_local void @f3()
  f1();
  // LINUX: call win64cc void @f1()
  // WIN64: call void @f1()
  f2();
  // COMMON: call void @f2()
}
// LINUX: declare win64cc void @f1()
// LINUX: declare void @f2()
// WIN64: declare dso_local void @f1()
// WIN64: declare dso_local void @f2()

// Win64 ABI varargs
void __attribute__((ms_abi)) f4(int a, ...) {
  // LINUX-LABEL: define{{.*}} win64cc void @f4
  // WIN64-LABEL: define dso_local void @f4
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // COMMON: %[[AP:.*]] = alloca i8*
  // COMMON: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // COMMON: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // COMMON-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // COMMON-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  __builtin_ms_va_list ap2;
  __builtin_ms_va_copy(ap2, ap);
  // COMMON: %[[AP_VAL:.*]] = load i8*, i8** %[[AP]]
  // COMMON-NEXT: store i8* %[[AP_VAL]], i8** %[[AP2:.*]]
  __builtin_ms_va_end(ap);
  // COMMON: call void @llvm.va_end
}

void __attribute__((ms_abi)) f4_2(int a, ...) {
  // LINUX-LABEL: define{{.*}} win64cc void @f4_2
  // WIN64-LABEL: define dso_local void @f4_2
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // COMMON: %[[AP:.*]] = alloca i8*
  // COMMON: call void @llvm.va_start
  struct small_odd s1 = __builtin_va_arg(ap, struct small_odd);
  // COMMON: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // COMMON-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // COMMON-NEXT: bitcast i8* %[[AP_CUR]] to %struct.small_odd*
  struct larger s2 = __builtin_va_arg(ap, struct larger);
  // COMMON: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 8
  // COMMON-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // COMMON-NEXT: bitcast i8* %[[AP_CUR2]] to %struct.larger**
  __builtin_ms_va_end(ap);
}

// Let's verify that normal va_lists work right on Win64, too.
void f5(int a, ...) {
  // WIN64-LABEL: define dso_local void @f5
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  // WIN64: %[[AP:.*]] = alloca i8*
  // WIN64: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // WIN64: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  // WIN64: call void @llvm.va_copy
  __builtin_va_end(ap);
  // WIN64: call void @llvm.va_end
}
