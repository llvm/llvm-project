// Test the FIR types
// Parse types and check that we can reparse what we print.
// RUN: fir-opt %s | fir-opt | FileCheck %s

// Fortran Intrinsic types
// CHECK-LABEL: func private @it1() -> !fir.int<4>
// CHECK-LABEL: func private @it2() -> !fir.real<8>
// CHECK-LABEL: func private @it3() -> !fir.complex<8>
// CHECK-LABEL: func private @it4() -> !fir.logical<1>
// CHECK-LABEL: func private @it5() -> !fir.char<1>
// CHECK-LABEL: func private @it6() -> !fir.char<2,10>
// CHECK-LABEL: func private @it7() -> !fir.char<4,?>
func private @it1() -> !fir.int<4>
func private @it2() -> !fir.real<8>
func private @it3() -> !fir.complex<8>
func private @it4() -> !fir.logical<1>
func private @it5() -> !fir.char<1>
func private @it6() -> !fir.char<2,10>
func private @it7() -> !fir.char<4,?>

// Fortran Derived types (records)
// CHECK-LABEL: func private @dvd1() -> !fir.type<derived1>
// CHECK-LABEL: func private @dvd2() -> !fir.type<derived2(p:i32)>
// CHECK-LABEL: func private @dvd3() -> !fir.type<derived3{f:f32}>
// CHECK-LABEL: func private @dvd4() -> !fir.type<derived4(p:i8){f:f32}>
// CHECK-LABEL: func private @dvd5() -> !fir.type<derived5(p1:i8,p2:i8,p3:i8,p4:i8,p5:i8){f1:f32,f2:f32,f3:f32,f4:f32,f5:f32,f6:f32,f7:f32,f8:f32}>
// CHECK-LABEL: func private @dvd6() -> !fir.type<derived6{f:!fir.ptr<!fir.type<derived6>>}>
func private @dvd1() -> !fir.type<derived1>
func private @dvd2() -> !fir.type<derived2(p:i32)>
func private @dvd3() -> !fir.type<derived3{f:f32}>
func private @dvd4() -> !fir.type<derived4(p:i8){f:f32}>
func private @dvd5() -> !fir.type<derived5(p1:i8,p2:i8,p3:i8,p4:i8,p5:i8){f1:f32,f2:f32,f3:f32,f4:f32,f5:f32,f6:f32,f7:f32,f8:f32}>
func private @dvd6() -> !fir.type<derived6{f:!fir.ptr<!fir.type<derived6>>}>

// FIR array types
// CHECK-LABEL: func private @arr1() -> !fir.array<10xf32>
// CHECK-LABEL: func private @arr2() -> !fir.array<10x10xf32>
// CHECK-LABEL: func private @arr3() -> !fir.array<?xf32>
// CHECK-LABEL: func private @arr4() -> !fir.array<10x?xf32>
// CHECK-LABEL: func private @arr5() -> !fir.array<?x?xf32>
// CHECK-LABEL: func private @arr6() -> !fir.array<*:f32>
// CHECK-LABEL: func private @arr7() -> !fir.array<1x2x?x4x5x6x7x8x9xf32>
func private @arr1() -> !fir.array<10xf32>
func private @arr2() -> !fir.array<10x10xf32>
func private @arr3() -> !fir.array<?xf32>
func private @arr4() -> !fir.array<10x?xf32>
func private @arr5() -> !fir.array<?x?xf32>
func private @arr6() -> !fir.array<*:f32>
func private @arr7() -> !fir.array<1x2x?x4x5x6x7x8x9xf32>

// FIR pointer-like types
// CHECK-LABEL: func private @mem1() -> !fir.ref<i32>
// CHECK-LABEL: func private @mem2() -> !fir.ptr<i32>
// CHECK-LABEL: func private @mem3() -> !fir.heap<i32>
// CHECK-LABEL: func private @mem4() -> !fir.ref<() -> ()>
func private @mem1() -> !fir.ref<i32>
func private @mem2() -> !fir.ptr<i32>
func private @mem3() -> !fir.heap<i32>
func private @mem4() -> !fir.ref<() -> ()>

// FIR box types (descriptors)
// CHECK-LABEL: func private @box1() -> !fir.box<!fir.array<?xf32>>
// CHECK-LABEL: func private @box2() -> !fir.boxchar<2>
// CHECK-LABEL: func private @box3() -> !fir.boxproc<(i32, i32) -> i64>
// CHECK-LABEL: func private @box4() -> !fir.box<none>
// CHECK-LABEL: func private @box5() -> !fir.box<!fir.type<derived3{f:f32}>>
func private @box1() -> !fir.box<!fir.array<?xf32>>
func private @box2() -> !fir.boxchar<2>
func private @box3() -> !fir.boxproc<(i32, i32) -> i64>
func private @box4() -> !fir.box<none>
func private @box5() -> !fir.box<!fir.type<derived3{f:f32}>>

// FIR misc. types
// CHECK-LABEL: func private @oth1() -> !fir.shape<1>
// CHECK-LABEL: func private @oth2() -> !fir.field
// CHECK-LABEL: func private @oth3() -> !fir.tdesc<!fir.type<derived7{f1:f32,f2:f32}>>
// CHECK-LABEL: func private @oth4() -> !fir.shapeshift<15>
// CHECK-LABEL: func private @oth5() -> !fir.slice<8>
func private @oth1() -> !fir.shape<1>
func private @oth2() -> !fir.field
func private @oth3() -> !fir.tdesc<!fir.type<derived7{f1:f32,f2:f32}>>
func private @oth4() -> !fir.shapeshift<15>
func private @oth5() -> !fir.slice<8>

// FIR vector
// CHECK-LABEL: func private @vecty(i1) -> !fir.vector<10:i32>
func private @vecty(i1) -> !fir.vector<10:i32>
