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
  // COMMON: %[[AP:.*]] = alloca ptr
  // COMMON: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // COMMON: %[[AP_CUR:.*]] = load ptr, ptr %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR]], i64 8
  // COMMON-NEXT: store ptr %[[AP_NEXT]], ptr %[[AP]]
  __builtin_ms_va_list ap2;
  __builtin_ms_va_copy(ap2, ap);
  // COMMON: %[[AP_VAL:.*]] = load ptr, ptr %[[AP]]
  // COMMON-NEXT: store ptr %[[AP_VAL]], ptr %[[AP2:.*]]
  __builtin_ms_va_end(ap);
  // COMMON: call void @llvm.va_end
}

void __attribute__((ms_abi)) f4_2(int a, ...) {
  // LINUX-LABEL: define{{.*}} win64cc void @f4_2
  // WIN64-LABEL: define dso_local void @f4_2
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // COMMON: %[[AP:.*]] = alloca ptr
  // COMMON: call void @llvm.va_start
  struct small_odd s1 = __builtin_va_arg(ap, struct small_odd);
  // COMMON: %[[AP_CUR:.*]] = load ptr, ptr %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR]], i64 8
  // COMMON-NEXT: store ptr %[[AP_NEXT]], ptr %[[AP]]
  struct larger s2 = __builtin_va_arg(ap, struct larger);
  // COMMON: %[[AP_CUR2:.*]] = load ptr, ptr %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT3:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR2]], i64 8
  // COMMON-NEXT: store ptr %[[AP_NEXT3]], ptr %[[AP]]
  __builtin_ms_va_end(ap);
}

// Let's verify that normal va_lists work right on Win64, too.
void f5(int a, ...) {
  // WIN64-LABEL: define dso_local void @f5
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  // WIN64: %[[AP:.*]] = alloca ptr
  // WIN64: call void @llvm.va_start
  int b = __builtin_va_arg(ap, int);
  // WIN64: %[[AP_CUR:.*]] = load ptr, ptr %[[AP]]
  // WIN64-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR]], i64 8
  // WIN64-NEXT: store ptr %[[AP_NEXT]], ptr %[[AP]]
  __builtin_va_list ap2;
  __builtin_va_copy(ap2, ap);
  // WIN64: call void @llvm.va_copy
  __builtin_va_end(ap);
  // WIN64: call void @llvm.va_end
}

struct HFA {
  float a, b, c;
};

__attribute__((ms_abi)) void msabi_hfa(struct HFA a);
__attribute__((ms_abi)) void msabi_hfa_vararg(struct HFA a, int b, ...);

void call_msabi_hfa(void) {
  // COMMON-LABEL: define{{.*}} void @call_msabi_hfa()
  // WIN64: call void @msabi_hfa([3 x float] {{.*}})
  // LINUX: call win64cc void @msabi_hfa([3 x float] {{.*}})
  msabi_hfa((struct HFA){1.0f, 2.0f, 3.0f});
}

void call_msabi_hfa_vararg(void) {
  // COMMON-LABEL: define{{.*}} void @call_msabi_hfa_vararg()
  // WIN64: call void ([2 x i64], i32, ...) @msabi_hfa_vararg([2 x i64] {{.*}}, i32 noundef 4, [2 x i64] {{.*}})
  // LINUX: call win64cc void ([2 x i64], i32, ...) @msabi_hfa_vararg([2 x i64] {{.*}}, i32 noundef 4, [2 x i64] {{.*}})
  msabi_hfa_vararg((struct HFA){1.0f, 2.0f, 3.0f}, 4,
                   (struct HFA){5.0f, 6.0f, 7.0f});
}

__attribute__((ms_abi)) void get_msabi_hfa_vararg(int a, ...) {
  // COMMON-LABEL: define{{.*}} void @get_msabi_hfa_vararg
  __builtin_ms_va_list ap;
  __builtin_ms_va_start(ap, a);
  // COMMON: %[[AP:.*]] = alloca ptr
  // COMMON: call void @llvm.va_start
  struct HFA b = __builtin_va_arg(ap, struct HFA);
  // COMMON: %[[AP_CUR:.*]] = load ptr, ptr %[[AP]]
  // COMMON-NEXT: %[[AP_NEXT:.*]] = getelementptr inbounds i8, ptr %[[AP_CUR]], i64 16
  // COMMON-NEXT: store ptr %[[AP_NEXT]], ptr %[[AP]]
  __builtin_ms_va_end(ap);
}
