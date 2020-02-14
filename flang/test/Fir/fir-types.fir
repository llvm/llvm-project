// Test the FIR types

// RUN: tco -emit-fir %s | tco -emit-fir | FileCheck %s
// UNSUPPORTED: !fir

// Fortran Intrinsic types
// CHECK-LABEL: func @it1() -> !fir.int<4>
// CHECK-LABEL: func @it2() -> !fir.real<8>
// CHECK-LABEL: func @it3() -> !fir.complex<8>
// CHECK-LABEL: func @it4() -> !fir.logical<1>
// CHECK-LABEL: func @it5() -> !fir.char<1>
func @it1() -> !fir.int<4>
func @it2() -> !fir.real<8>
func @it3() -> !fir.complex<8>
func @it4() -> !fir.logical<1>
func @it5() -> !fir.char<1>

// Fortran Derived types (records)
// CHECK-LABEL: func @dvd1() -> !fir.type<derived1>
// CHECK-LABEL: func @dvd2() -> !fir.type<derived2(p:i32)>
// CHECK-LABEL: func @dvd3() -> !fir.type<derived3{f:f32}>
// CHECK-LABEL: func @dvd4() -> !fir.type<derived4(p:i8){f:f32}>
// CHECK-LABEL: func @dvd5() -> !fir.type<derived5(p1:i8,p2:i8,p3:i8,p4:i8,p5:i8){f1:f32,f2:f32,f3:f32,f4:f32,f5:f32,f6:f32,f7:f32,f8:f32}>
// CHECK-LABEL: func @dvd6() -> !fir.type<derived6{f:!fir.ptr<!fir.type<derived6>>}>
func @dvd1() -> !fir.type<derived1>
func @dvd2() -> !fir.type<derived2(p:i32)>
func @dvd3() -> !fir.type<derived3{f:f32}>
func @dvd4() -> !fir.type<derived4(p:i8){f:f32}>
func @dvd5() -> !fir.type<derived5(p1:i8,p2:i8,p3:i8,p4:i8,p5:i8){f1:f32,f2:f32,f3:f32,f4:f32,f5:f32,f6:f32,f7:f32,f8:f32}>
func @dvd6() -> !fir.type<derived6{f:!fir.ptr<!fir.type<derived6>>}>

// FIR array types
// CHECK-LABEL: func @arr1() -> !fir.array<10xf32>
// CHECK-LABEL: func @arr2() -> !fir.array<10x10xf32>
// CHECK-LABEL: func @arr3() -> !fir.array<?xf32>
// CHECK-LABEL: func @arr4() -> !fir.array<10x?xf32>
// CHECK-LABEL: func @arr5() -> !fir.array<?x?xf32>
// CHECK-LABEL: func @arr6() -> !fir.array<*:f32>
// CHECK-LABEL: func @arr7() -> !fir.array<1x2x?x4x5x6x7x8x9xf32>
func @arr1() -> !fir.array<10xf32>
func @arr2() -> !fir.array<10x10xf32>
func @arr3() -> !fir.array<?xf32>
func @arr4() -> !fir.array<10x?xf32>
func @arr5() -> !fir.array<?x?xf32>
func @arr6() -> !fir.array<*:f32>
func @arr7() -> !fir.array<1x2x?x4x5x6x7x8x9xf32>

// FIR pointer-like types
// CHECK-LABEL: func @mem1() -> !fir.ref<i32>
// CHECK-LABEL: func @mem2() -> !fir.ptr<i32>
// CHECK-LABEL: func @mem3() -> !fir.heap<i32>
// CHECK-LABEL: func @mem4() -> !fir.ref<() -> ()>
func @mem1() -> !fir.ref<i32>
func @mem2() -> !fir.ptr<i32>
func @mem3() -> !fir.heap<i32>
func @mem4() -> !fir.ref<() -> ()>

// FIR box types (descriptors)
// CHECK-LABEL: func @box1() -> !fir.box<!fir.array<?xf32>>
// CHECK-LABEL: func @box2() -> !fir.boxchar<2>
// CHECK-LABEL: func @box3() -> !fir.boxproc<(i32, i32) -> i64>
// CHECK-LABEL: func @box4() -> !fir.box<none>
// CHECK-LABEL: func @box5() -> !fir.box<!fir.type<derived3{f:f32}>>
func @box1() -> !fir.box<!fir.array<?xf32>>
func @box2() -> !fir.boxchar<2>
func @box3() -> !fir.boxproc<(i32, i32) -> i64>
func @box4() -> !fir.box<none>
func @box5() -> !fir.box<!fir.type<derived3{f:f32}>>

// FIR misc. types
// CHECK-LABEL: func @oth1() -> !fir.dims<1>
// CHECK-LABEL: func @oth2() -> !fir.field
// CHECK-LABEL: func @oth3() -> !fir.tdesc<!fir.type<derived7{f1:f32,f2:f32}>>
// CHECK-LABEL: func @oth4() -> !fir.dims<15>
func @oth1() -> !fir.dims<1>
func @oth2() -> !fir.field
func @oth3() -> !fir.tdesc<!fir.type<derived7{f1:f32,f2:f32}>>
func @oth4() -> !fir.dims<15>
