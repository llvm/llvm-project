// RUN: %clang_cc1 %s -std=c++1y -triple=i686-pc-win32 -emit-llvm -o - | FileCheck %s

struct A {
  A();
  ~A();
};

// CHECK-DAG: $"\01??$a@X@@3UA@@A" = comdat any
// CHECK-DAG: @"\01??$a@X@@3UA@@A" = linkonce_odr thread_local global %struct.A zeroinitializer, comdat, align 1
// CHECK-DAG: @"\01??__E?$a@X@@YAXXZ$initializer$" = internal constant void ()* @"\01??__E?$a@X@@YAXXZ", section ".CRT$XDU", comdat($"\01??$a@X@@3UA@@A")
template <typename T>
thread_local A a = A();

// CHECK-DAG: @"\01?b@@3UA@@A" = thread_local global %struct.A zeroinitializer, align 1
// CHECK-DAG: @"__tls_init$initializer$" = internal constant void ()* @__tls_init, section ".CRT$XDU"
thread_local A b;

// CHECK-LABEL: define internal void @__tls_init()
// CHECK: call void @"\01??__Eb@@YAXXZ"

thread_local A &c = b;
thread_local A &d = c;

A f() {
  (void)a<void>;
  (void)b;
  return c;
}

// CHECK: !llvm.module.flags = !{{{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 6, !"Linker Options", ![[link_opts:[0-9]+]]}
// CHECK: ![[link_opts]] = !{![[dyn_tls_init:[0-9]+]]}
// CHECK: ![[dyn_tls_init]] = !{!"/include:___dyn_tls_init@12"}
