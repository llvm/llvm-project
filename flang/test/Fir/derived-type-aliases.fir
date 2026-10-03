// RUN: fir-opt %s | FileCheck %s --check-prefix=DEFAULT
// RUN: fir-opt -enable-fir-type-aliases %s | FileCheck %s --check-prefix=ALIASES

module {
  func.func @use_aliases(
      %arg0: !fir.ref<!fir.type<outer{inner:!fir.type<inner{p:!fir.box<!fir.ptr<f32>>}>}>>,
      %arg1: !fir.ref<!fir.type<a.b{x:i32}>>,
      %arg2: !fir.ref<!fir.type<a_b{x:i64}>>,
      %arg3: !fir.ref<!fir.type<recursive{next:!fir.box<!fir.heap<!fir.type<recursive>>>}>>) {
    return
  }
}

// DEFAULT-NOT: !outer =
// DEFAULT-NOT: !aXb =
// DEFAULT-NOT: !a_b =
// DEFAULT: func.func @use_aliases(%{{.*}}: !fir.ref<!fir.type<outer{inner:!fir.type<inner{p:!fir.box<!fir.ptr<f32>>}>}>>, %{{.*}}: !fir.ref<!fir.type<a.b{x:i32}>>, %{{.*}}: !fir.ref<!fir.type<a_b{x:i64}>>, %{{.*}}: !fir.ref<!fir.type<recursive{next:!fir.box<!fir.heap<!fir.type<recursive>>>}>>)

// ALIASES-DAG: !aXb = !fir.type<a.b{x:i32}>
// ALIASES-DAG: !a_b = !fir.type<a_b{x:i64}>
// ALIASES-DAG: !inner = !fir.type<inner{p:!fir.box<!fir.ptr<f32>>}>
// ALIASES-DAG: !outer = !fir.type<outer{inner:!inner}>
// ALIASES-DAG: !recursive = !fir.type<recursive{next:!fir.box<!fir.heap<!fir.type<recursive>>>}>
// ALIASES: func.func @use_aliases(%{{.*}}: !fir.ref<!outer>, %{{.*}}: !fir.ref<!aXb>, %{{.*}}: !fir.ref<!a_b>, %{{.*}}: !fir.ref<!recursive>)
