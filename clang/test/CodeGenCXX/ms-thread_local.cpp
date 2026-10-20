// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -fms-compatibility-version=19.25 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -fms-compatibility-version=19.20 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LEGACY
// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -fms-compatibility-version=19.25 -fno-ms-tls-guards -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-NO-GUARDS
// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -fms-compatibility-version=19.25 -ftls-model=local-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LD

struct A {
  A();
  ~A();
};

// CHECK-DAG: $"??$a@X@@3UA@@A" = comdat any
// CHECK-DAG: @"??$a@X@@3UA@@A" = linkonce_odr dso_local thread_local global %struct.A zeroinitializer, comdat, align 1
// CHECK-DAG: @"??__E?$a@X@@YAXXZ$initializer$" = internal constant ptr @"??__E?$a@X@@YAXXZ", section ".CRT$XDU", comdat($"??$a@X@@3UA@@A")
// CHECK-LD-DAG: $"??$a@X@@3UA@@A" = comdat any
// CHECK-LD-DAG: @"??$a@X@@3UA@@A" = linkonce_odr dso_local thread_local(localdynamic) global %struct.A zeroinitializer, comdat, align 1
// CHECK-LD-DAG: @"??__E?$a@X@@YAXXZ$initializer$" = internal constant ptr @"??__E?$a@X@@YAXXZ", section ".CRT$XDU", comdat($"??$a@X@@3UA@@A")
template <typename T>
thread_local A a = A();

// CHECK-DAG: @"?b@@3UA@@A" = dso_local thread_local global %struct.A zeroinitializer, align 1
// CHECK-DAG: @"__tls_init$initializer$" = internal constant ptr @__tls_init, section ".CRT$XDU"
// CHECK-DAG: @__tls_guard = external dso_local thread_local global i8
// CHECK-LD-DAG: @"?b@@3UA@@A" = dso_local thread_local(localdynamic) global %struct.A zeroinitializer, align 1
// CHECK-LD-DAG: @"__tls_init$initializer$" = internal constant ptr @__tls_init, section ".CRT$XDU"
// CHECK-LD-DAG: @__tls_guard = external dso_local thread_local global i8
// CHECK-LEGACY-NOT: @__tls_guard = external dso_local thread_local global i8
// CHECK-NO-GUARDS-NOT: @__tls_guard = external dso_local thread_local global i8
thread_local A b;

// CHECK-LABEL: declare dso_local void @__dyn_tls_on_demand_init()
// CHECK-LD-LABEL: declare dso_local void @__dyn_tls_on_demand_init()
// CHECK-LEGACY-NOT: declare dso_local void @__dyn_tls_on_demand_init()
// CHECK-NO-GUARDS-NOT: declare dso_local void @__dyn_tls_on_demand_init()

// CHECK-LABEL: define dso_local void @"?f@@YA?AUA@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.A) align 1 %agg.result)
// CHECK: call void @__dyn_tls_on_demand_init()
// CHECK-LD-LABEL: define dso_local void @"?f@@YA?AUA@@XZ"(ptr dead_on_unwind noalias writable sret(%struct.A) align 1 %agg.result)
// CHECK-LD: call void @__dyn_tls_on_demand_init()
// CHECK-LEGACY-NOT: call void @__dyn_tls_on_demand_init()
// CHECK-NO-GUARDS-NOT: call void @__dyn_tls_on_demand_init()

thread_local A &c = b;
thread_local A &d = c;

A f() {
  (void)a<void>;
  (void)b;
  return c;
}

// CHECK-LABEL: define dso_local noundef i32 @"?g@@YAHXZ"()
// CHECK-NOT: call void @__dyn_tls_on_demand_init()
// CHECK-LD-LABEL: define dso_local noundef i32 @"?g@@YAHXZ"()
// CHECK-LD-NOT: call void @__dyn_tls_on_demand_init()

thread_local int e = 2;

int g() {
  return e;
}

// CHECK-LABEL: define internal void @__tls_init()
// CHECK: call void @"??__Eb@@YAXXZ"
// CHECK-LD-LABEL: define internal void @__tls_init()
// CHECK-LD: call void @"??__Eb@@YAXXZ"

// CHECK: !llvm.linker.options = !{![[dyn_tls_init:[0-9]+]]}
// CHECK: ![[dyn_tls_init]] = !{!"/include:___dyn_tls_init@12"}
// CHECK-LD: !llvm.linker.options = !{![[dyn_tls_init:[0-9]+]]}
// CHECK-LD: ![[dyn_tls_init]] = !{!"/include:___dyn_tls_init@12"}
