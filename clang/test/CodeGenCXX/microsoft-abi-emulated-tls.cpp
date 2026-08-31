// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++11 \
// RUN:   -fms-compatibility-version=19.25 -femulated-tls \
// RUN:   -emit-llvm -o - %s | FileCheck \
// RUN:   --implicit-check-not=_Init_thread_epoch \
// RUN:   --implicit-check-not=__tls_guard %s

int make_value();

// CHECK-DAG: @"__tls_init$initializer$" = internal constant ptr @__tls_init, section ".CRT$XDU"
// CHECK-LABEL: define dso_local noundef i32 @"?guarded_value@@YAHXZ"()
// CHECK: br i1 true, label %[[ATTEMPT:[a-z.]+]], label %[[END:[a-z.]+]]
// CHECK: [[ATTEMPT]]:
// CHECK: call void @_Init_thread_header
// CHECK: call void @_Init_thread_footer
// CHECK: [[END]]:
int guarded_value() {
  static int value = make_value();
  return value;
}

int make_tls_value();
thread_local int dynamic_tls = make_tls_value();

// CHECK-LABEL: define dso_local noundef i32 @"?read_dynamic_tls@@YAHXZ"()
// CHECK-NEXT: entry:
// CHECK-NEXT: call void @__dyn_tls_on_demand_init()
// CHECK: call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @"?dynamic_tls@@3HA")
int read_dynamic_tls() { return dynamic_tls; }
