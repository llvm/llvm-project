// RUN: %clang_cc1 -triple x86_64-unknown-freebsd10.0 -emit-llvm < %s | FileCheck -check-prefix=FREEBSD %s
// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm < %s | FileCheck -check-prefix=WIN64 %s

struct foo {
  int x;
  float y;
  char z;
};
// FREEBSD: %[[STRUCT_FOO:.*]] = type { i32, float, i8 }
// WIN64: %[[STRUCT_FOO:.*]] = type { i32, float, i8 }

void __attribute__((ms_abi)) f1(void);
void __attribute__((sysv_abi)) f2(void);
void f3(void) {
  // FREEBSD-LABEL: define void @f3()
  // WIN64-LABEL: define dso_local void @f3()
  f1();
  // FREEBSD: call win64cc void @f1()
  // WIN64: call void @f1()
  f2();
  // FREEBSD: call void @f2()
  // WIN64: call x86_64_sysvcc void @f2()
}
// FREEBSD: declare win64cc void @f1()
// FREEBSD: declare void @f2()
// WIN64: declare dso_local void @f1()
// WIN64: declare dso_local x86_64_sysvcc void @f2()

// Win64 ABI varargs
void __attribute__((ms_abi)) f4(int a, ...) {
  // FREEBSD-LABEL: define win64cc void @f4
  // WIN64-LABEL: define dso_local void @f4
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // FREEBSD: %[[AP:.*]] = alloca i8*
  // FREEBSD: call void @llvm.va_start
  // WIN64: %[[AP:.*]] = alloca i8*
  // WIN64: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // FREEBSD: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // FREEBSD-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  // WIN64: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  // FIXME: These are different now. We probably need __builtin_ms_va_arg.
  double _Complex c = __builtin_va_arg(ap, double _Complex);
  // FREEBSD: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 16
  // FREEBSD-NEXT: store i8* %[[AP_NEXT2]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR2]] to { double, double }*
  // WIN64: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT2]], i8** %[[AP]]
  // WIN64-NEXT: %[[CUR2:.*]] = bitcast i8* %[[AP_CUR2]] to { double, double }**
  // WIN64-NEXT: load { double, double }*, { double, double }** %[[CUR2]]
  struct foo d = __builtin_va_arg(ap, struct foo);
  // FREEBSD: %[[AP_CUR3:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR3]], i64 16
  // FREEBSD-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR3]] to %[[STRUCT_FOO]]*
  // WIN64: %[[AP_CUR3:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR3]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // WIN64-NEXT: %[[CUR3:.*]] = bitcast i8* %[[AP_CUR3]] to %[[STRUCT_FOO]]*
  // WIN64-NEXT: load %[[STRUCT_FOO]]*, %[[STRUCT_FOO]]** %[[CUR3]]
  __builtin_ms_va_list ap2;
  __builtin_ms_va_copy(ap2, ap);
  // FREEBSD: %[[AP_VAL:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: store i8* %[[AP_VAL]], i8** %[[AP2:.*]]
  // WIN64: %[[AP_VAL:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: store i8* %[[AP_VAL]], i8** %[[AP2:.*]]
  __builtin_ms_va_end(ap);
  // FREEBSD: call void @llvm.va_end
  // WIN64: call void @llvm.va_end
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
  double _Complex c = __builtin_va_arg(ap, double _Complex);
  // WIN64: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT2]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR2]] to { double, double }*
  struct foo d = __builtin_va_arg(ap, struct foo);
  // WIN64: %[[AP_CUR3:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR3]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR3]] to %[[STRUCT_FOO]]*
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  // WIN64: call void @llvm.va_copy
  __builtin_va_end(ap);
  // WIN64: call void @llvm.va_end
}

// Verify that using a Win64 va_list from a System V function works.
void __attribute__((sysv_abi)) f6(__builtin_ms_va_list ap) {
  // FREEBSD-LABEL: define void @f6
  // FREEBSD: store i8* %ap, i8** %[[AP:.*]]
  // WIN64-LABEL: define dso_local x86_64_sysvcc void @f6
  // WIN64: store i8* %ap, i8** %[[AP:.*]]
  int b = __builtin_va_arg(ap, int);
  // FREEBSD: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // FREEBSD-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  // WIN64: %[[AP_CUR:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR]] to i32*
  double _Complex c = __builtin_va_arg(ap, double _Complex);
  // FREEBSD: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 16
  // FREEBSD-NEXT: store i8* %[[AP_NEXT2]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR2]] to { double, double }*
  // WIN64: %[[AP_CUR2:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT2:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR2]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT2]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR2]] to { double, double }*
  struct foo d = __builtin_va_arg(ap, struct foo);
  // FREEBSD: %[[AP_CUR3:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR3]], i64 16
  // FREEBSD-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // FREEBSD-NEXT: bitcast i8* %[[AP_CUR3]] to %[[STRUCT_FOO]]*
  // WIN64: %[[AP_CUR3:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, i8* %[[AP_CUR3]], i64 8
  // WIN64-NEXT: store i8* %[[AP_NEXT3]], i8** %[[AP]]
  // WIN64-NEXT: bitcast i8* %[[AP_CUR3]] to %[[STRUCT_FOO]]*
  __builtin_ms_va_list ap2;
  __builtin_ms_va_copy(ap2, ap);
  // FREEBSD: %[[AP_VAL:.*]] = load i8*, i8** %[[AP]]
  // FREEBSD-NEXT: store i8* %[[AP_VAL]], i8** %[[AP2:.*]]
  // WIN64: %[[AP_VAL:.*]] = load i8*, i8** %[[AP]]
  // WIN64-NEXT: store i8* %[[AP_VAL]], i8** %[[AP2:.*]]
}

// This test checks if structs are passed according to Win64 calling convention
// when it's enforced by __attribute((ms_abi)).
struct i128 {
  unsigned long long a;
  unsigned long long b;
};

__attribute__((ms_abi)) struct i128 f7(struct i128 a) {
  // WIN64: define dso_local void @f7(%struct.i128* noalias sret(%struct.i128) align 8 %agg.result, %struct.i128* %a)
  // FREEBSD: define win64cc void @f7(%struct.i128* noalias sret(%struct.i128) align 8 %agg.result, %struct.i128* %a)
  return a;
}
