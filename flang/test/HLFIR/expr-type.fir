// Test the HLFIR Expr type
// Parse types and check that we can reparse what we print.
// RUN: fir-opt --split-input-file %s | fir-opt --split-input-file | FileCheck %s

// Scalar expression types
func.func private @scalar01() -> !hlfir.expr<!fir.char<1,?>>
func.func private @scalar02() -> !hlfir.expr<!fir.char<2,10>>
func.func private @scalar03() -> !hlfir.expr<!fir.type<derived{f:f32}>>
// CHECK-LABEL:  func.func private @scalar01() -> !hlfir.expr<!fir.char<1,?>>
// CHECK-LABEL:  func.func private @scalar02() -> !hlfir.expr<!fir.char<2,10>>
// CHECK-LABEL:  func.func private @scalar03() -> !hlfir.expr<!fir.type<derived{f:f32}>>

// Array expression types
func.func private @array01() -> !hlfir.expr<?x!fir.char<1,?>>
func.func private @array02() -> !hlfir.expr<10x!fir.char<1,20>>
func.func private @array03() -> !hlfir.expr<10x?x20x?x30x?x40x?x50x?x60xf128>
func.func private @array04() -> !hlfir.expr<10x20x!fir.type<derived{f:f32}>>
func.func private @array05() -> !hlfir.expr<10xi32>
// CHECK-LABEL:  func.func private @array01() -> !hlfir.expr<?x!fir.char<1,?>>
// CHECK-LABEL:  func.func private @array02() -> !hlfir.expr<10x!fir.char<1,20>>
// CHECK-LABEL:  func.func private @array03() -> !hlfir.expr<10x?x20x?x30x?x40x?x50x?x60xf128>
// CHECK-LABEL:  func.func private @array04() -> !hlfir.expr<10x20x!fir.type<derived{f:f32}>>
// CHECK-LABEL:  func.func private @array05() -> !hlfir.expr<10xi32>

// Polymorphic expression types
func.func private @polymorph01() -> !hlfir.expr<!fir.type<derived{f:f32}>?>
func.func private @polymorph02() -> !hlfir.expr<?x!fir.type<derived{f:f32}>?>
func.func private @polymorph03() -> !hlfir.expr<10x!fir.type<derived{f:f32}>?>
// CHECK-LABEL:  func.func private @polymorph01() -> !hlfir.expr<!fir.type<derived{f:f32}>?>
// CHECK-LABEL:  func.func private @polymorph02() -> !hlfir.expr<?x!fir.type<derived{f:f32}>?>
// CHECK-LABEL:  func.func private @polymorph03() -> !hlfir.expr<10x!fir.type<derived{f:f32}>?>
