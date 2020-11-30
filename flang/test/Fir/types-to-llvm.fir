// RUN: tco %s | FileCheck %s

// CHECK-LABEL: declare void @foo0([20 x [10 x [5 x i32]]]*)
func private @foo0(%arg0: !fir.ref<!fir.array<5x10x20xi32>>)
// CHECK-LABEL: declare void @foo1(i32*)
func private @foo1(%arg0: !fir.ref<!fir.array<?x10x20xi32>>)
// CHECK-LABEL: declare void @foo2(i32*)
func private @foo2(%arg0: !fir.ref<!fir.array<5x?x20xi32>>)
// CHECK-LABEL: declare void @foo3([10 x [5 x i32]]*)
func private @foo3(%arg0: !fir.ref<!fir.array<5x10x?xi32>>)
// CHECK-LABEL: declare void @foo4(i32*)
func private @foo4(%arg0: !fir.ref<!fir.array<?x?x?xi32>>)

// CHECK-LABEL: declare void @byval5(i32*)
func private @byval5(%arg0: !fir.array<?x?x?xi32>)
