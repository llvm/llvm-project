// Test FIR types conversion.

// RUN: fir-opt --split-input-file --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s


// Test sequence types `!fir.array`
// Note that we're switching from column-major to row-major here.

func.func private @foo0(%arg0: !fir.array<10x12xi64>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.array<12 x array<10 x i64>>
func.func private @foo1(%arg0: !fir.array<?xf32>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<f32>
func.func private @foo2(%arg0: !fir.array<?x?xf128>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.ptr<f128>
func.func private @foo3(%arg0: !fir.array<*:i32>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<i32>
func.func private @foo4(%arg0: !fir.array<3x?xi32>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.ptr<array<3 x i32>>
func.func private @foo5(%arg0: !fir.array<?x6xi32>)
// CHECK-LABEL: foo5
// CHECK-SAME: !llvm.ptr<i32>
func.func private @foo6(%arg0: !fir.array<2x?x3x?x4xi32>)
// CHECK-LABEL: foo6
// CHECK-SAME: !llvm.ptr<array<2 x i32>
func.func private @foo7(%arg0: !fir.array<6x?x!fir.char<1,?>>)
// CHECK-LABEL: foo7
// CHECK-SAME: !llvm.ptr<i8>
func.func private @foo8(%arg0: !fir.array<6x?x!fir.char<1,3>>)
// CHECK-LABEL: foo8
// CHECK-SAME: !llvm.ptr<array<6 x array<3 x i8>

// -----

// Test reference types `!fir.ref`

func.func private @foo0(%arg0: !fir.ref<i32>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<i32>
func.func private @foo1(%arg0: !fir.ref<!fir.array<10xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<array<10 x f32>>
func.func private @foo2(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMs1Ta1{x:i32,y:f32}>>>>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.ptr<struct<(ptr<struct<"_QMs1Ta1", (i32, f32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>>
func.func private @foo3(%arg0: !fir.ref<!fir.array<10x?x11xf32>>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<array<10 x f32>>
func.func private @foo4(%arg0: !fir.ref<!fir.array<10x11x?x?xf32>>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.ptr<array<11 x array<10 x f32>>>
func.func private @foo5(%arg0: !fir.ref<!fir.array<10x!fir.char<1,?>>>)
// CHECK-LABEL: foo5
// CHECK-SAME: !llvm.ptr<i8>
func.func private @foo6(%arg0: !fir.ref<!fir.array<10x!fir.char<2,8>>>)
// CHECK-LABEL: foo6
// CHECK-SAME: !llvm.ptr<array<10 x array<8 x i16>
func.func private @foo7(%arg0: !fir.ref<!fir.box<!fir.array<?xf32>>>)
// CHECK-LABEL: foo7
// CHECK-SAME: !llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i{{.*}}>>)>>

// -----

// Test pointer types `!fir.ptr`

func.func private @foo0(%arg0: !fir.ptr<i32>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<i32>

func.func private @foo1(%arg0: !fir.ptr<!fir.array<10xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<array<10 x f32>>

// -----

// Test box types `!fir.box`

func.func private @foo0(%arg0: !fir.box<!fir.array<?xf32>>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<struct<(ptr<f32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i{{.*}}>>)>>

func.func private @foo1(%arg0: !fir.box<!fir.array<10xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<struct<(ptr<array<10 x f32>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, array<1 x array<3 x i{{.*}}>>)>>

func.func private @foo2(%arg0: !fir.box<!fir.ref<i64>>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.ptr<struct<(ptr<i64>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>

func.func private @foo3(%arg0: !fir.box<!fir.type<derived{f:f32}>>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<struct<(ptr<struct<"derived", (f32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>>

func.func private @foo4(%arg0: !fir.box<!fir.heap<!fir.type<_QMs1Ta1{x:i32,y:f32}>>>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.ptr<struct<(ptr<struct<"_QMs1Ta1", (i32, f32)>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, ptr<i{{.*}}>, array<1 x i{{.*}}>)>>

// -----

// Test char types `!fir.char<k, n>`

func.func private @foo0(%arg0: !fir.char<1, 4>, %arg1: !fir.char<1, ?>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.array<4 x i8>
// CHECK-SAME: i8

func.func private @foo1(%arg0: !fir.char<2, 12>, %arg1: !fir.char<2, ?>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.array<12 x i16>
// CHECK-SAME: i16

func.func private @foo2(%arg0: !fir.char<4, 8>, %arg1: !fir.char<4, ?>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.array<8 x i32>
// CHECK-SAME: i32

// -----

// Test `!fir.heap<>` conversion.
func.func private @foo0(%arg0: !fir.heap<i32>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<i32>

func.func private @foo1(%arg0: !fir.heap<!fir.array<4xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<array<4 x f32>>

func.func private @foo2(%arg0: !fir.heap<!fir.array<?xf32>>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.ptr<f32>

func.func private @foo3(%arg0: !fir.heap<!fir.char<1,10>>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<array<10 x i8>>

func.func private @foo4(%arg0: !fir.heap<!fir.char<1,?>>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.ptr<i8>

func.func private @foo5(%arg0: !fir.heap<!fir.array<2xf32>>)
// CHECK-LABEL: foo5
// CHECK-SAME: !llvm.ptr<array<2 x f32>>

func.func private @foo6(%arg0: !fir.heap<!fir.array<?x?xf32>>)
// CHECK-LABEL: foo6
// CHECK-SAME: !llvm.ptr<f32>

func.func private @foo7(%arg0: !fir.heap<!fir.type<ZT>>)
// CHECK-LABEL: foo7
// CHECK-SAME: !llvm.ptr<struct<"ZT", ()>>

func.func private @foo8(%arg0: !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>)
// CHECK-LABEL: foo8
// CHECK-SAME: !llvm.ptr<struct<"_QMalloc_assignTt", (i32)>>

// -----

// Test `!fir.integer<KIND>` conversion.

func.func private @foo0(%arg0: !fir.int<1>)
// CHECK-LABEL: foo0
// CHECK-SAME: i8

func.func private @foo1(%arg0: !fir.int<2>)
// CHECK-LABEL: foo1
// CHECK-SAME: i16

func.func private @foo2(%arg0: !fir.int<4>)
// CHECK-LABEL: foo2
// CHECK-SAME: i32

func.func private @foo3(%arg0: !fir.int<8>)
// CHECK-LABEL: foo3
// CHECK-SAME: i64

func.func private @foo4(%arg0: !fir.int<16>)
// CHECK-LABEL: foo4
// CHECK-SAME: i128

// -----

// Test `!fir.logical<KIND>` conversion.

func.func private @foo0(%arg0: !fir.logical<1>)
// CHECK-LABEL: foo0
// CHECK-SAME: i8

func.func private @foo1(%arg0: !fir.logical<2>)
// CHECK-LABEL: foo1
// CHECK-SAME: i16

func.func private @foo2(%arg0: !fir.logical<4>)
// CHECK-LABEL: foo2
// CHECK-SAME: i32

func.func private @foo3(%arg0: !fir.logical<8>)
// CHECK-LABEL: foo3
// CHECK-SAME: i64

func.func private @foo4(%arg0: !fir.logical<16>)
// CHECK-LABEL: foo4
// CHECK-SAME: i128

// -----

// Test `!fir.llvm_ptr` conversion.

func.func private @foo0(%arg0: !fir.llvm_ptr<i8>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<i8>

func.func private @foo1(%arg0: !fir.llvm_ptr<!fir.ref<f32>>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.ptr<ptr<f32>>

func.func private @foo2(%arg0: !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<i32>>>>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.ptr<ptr<struct<(ptr<i32>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>>

func.func private @foo3(%arg0: !fir.llvm_ptr<!fir.ptr<f32>>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.ptr<ptr<f32>>

// -----

// Test MLIR `complex<KIND>` conversion.

func.func private @foo0(%arg0: complex<f16>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.struct<(f16, f16)>)

func.func private @foo1(%arg0: complex<bf16>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.struct<(bf16, bf16)>)

func.func private @foo2(%arg0: complex<f32>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.struct<(f32, f32)>)

func.func private @foo3(%arg0: complex<f64>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.struct<(f64, f64)>)

func.func private @foo4(%arg0: complex<f80>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.struct<(f80, f80)>)

func.func private @foo5(%arg0: complex<f128>)
// CHECK-LABEL: foo5
// CHECK-SAME: !llvm.struct<(f128, f128)>)

// -----

// Test `!fir.complex<KIND>` conversion.

func.func private @foo0(%arg0: !fir.complex<2>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.struct<(f16, f16)>)

func.func private @foo1(%arg0: !fir.complex<3>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.struct<(bf16, bf16)>)

func.func private @foo2(%arg0: !fir.complex<4>)
// CHECK-LABEL: foo2
// CHECK-SAME: !llvm.struct<(f32, f32)>)

func.func private @foo3(%arg0: !fir.complex<8>)
// CHECK-LABEL: foo3
// CHECK-SAME: !llvm.struct<(f64, f64)>)

func.func private @foo4(%arg0: !fir.complex<10>)
// CHECK-LABEL: foo4
// CHECK-SAME: !llvm.struct<(f80, f80)>)

func.func private @foo5(%arg0: !fir.complex<16>)
// CHECK-LABEL: foo5
// CHECK-SAME: !llvm.struct<(f128, f128)>)

// -----

// Test `!fir.vector<>` conversion.

func.func private @foo0(%arg0: !fir.vector<2:f16>)
// CHECK-LABEL: foo0
// CHECK-SAME: vector<2xf16>

func.func private @foo1(%arg0: !fir.vector<20:bf16>)
// CHECK-LABEL: foo1
// CHECK-SAME: vector<20xbf16>

func.func private @foo2(%arg0: !fir.vector<30:f32>)
// CHECK-LABEL: foo2
// CHECK-SAME: vector<30xf32>

func.func private @foo3(%arg0: !fir.vector<55:f64>)
// CHECK-LABEL: foo3
// CHECK-SAME: vector<55xf64>

func.func private @foo4(%arg0: !fir.vector<15:f80>)
// CHECK-LABEL: foo4
// CHECK-SAME: vector<15xf80>

func.func private @foo5(%arg0: !fir.vector<28:f128>)
// CHECK-LABEL: foo5
// CHECK-SAME: vector<28xf128>

func.func private @foo6(%arg0: !fir.vector<100:i1>)
// CHECK-LABEL: foo6
// CHECK-SAME: vector<100xi1>

func.func private @foo7(%arg0: !fir.vector<10:i8>)
// CHECK-LABEL: foo7
// CHECK-SAME: vector<10xi8>

func.func private @foo8(%arg0: !fir.vector<12:i16>)
// CHECK-LABEL: foo8
// CHECK-SAME: vector<12xi16>

func.func private @foo9(%arg0: !fir.vector<20:i32>)
// CHECK-LABEL: foo9
// CHECK-SAME: vector<20xi32>

func.func private @foo10(%arg0: !fir.vector<30:i64>)
// CHECK-LABEL: foo10
// CHECK-SAME: vector<30xi64>

func.func private @foo11(%arg0: !fir.vector<2:!fir.real<2>>)
// CHECK-LABEL: foo11
// CHECK-SAME: vector<2xf16>

func.func private @foo12(%arg0: !fir.vector<2:!fir.real<3>>)
// CHECK-LABEL: foo12
// CHECK-SAME: vector<2xbf16>

func.func private @foo13(%arg0: !fir.vector<2:!fir.real<4>>)
// CHECK-LABEL: foo13
// CHECK-SAME: vector<2xf32>

func.func private @foo14(%arg0: !fir.vector<2:!fir.real<8>>)
// CHECK-LABEL: foo14
// CHECK-SAME: vector<2xf64>

func.func private @foo15(%arg0: !fir.vector<2:!fir.real<10>>)
// CHECK-LABEL: foo15
// CHECK-SAME: vector<2xf80>

func.func private @foo16(%arg0: !fir.vector<2:!fir.real<16>>)
// CHECK-LABEL: foo16
// CHECK-SAME: vector<2xf128>

// -----

// Test `!fir.boxchar<n>` conversion

func.func private @foo0(%arg0: !fir.boxchar<1>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.struct<(ptr<i8>, i64)>

func.func private @foo1(%arg0: !fir.boxchar<2>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.struct<(ptr<i16>, i64)>

// -----

// Test `!fir.field` conversion

func.func private @foo0(%arg0: !fir.field)
// CHECK-LABEL: foo0
// CHECK-SAME: i32

// -----

// Test `!fir.len` conversion.

func.func private @foo0(%arg0: !fir.len)
// CHECK-LABEL: foo0
// CHECK-SAME: i64

// -----

// Test `!fir.tdesc` conversion.

func.func private @foo0(%arg0: !fir.tdesc<!fir.type<x>>)
// CHECK-LABEL: foo0
// CHECK-SAME:  !llvm.ptr<i8>

func.func private @foo1(%arg : !fir.tdesc<!fir.array<100xf32>>)
// CHECK-LABEL: foo1
// CHECK-SAME:  !llvm.ptr<i8>

func.func private @foo2(%arg : !fir.tdesc<f32>)
// CHECK-LABEL: foo2
// CHECK-SAME:  !llvm.ptr<i8>

func.func private @foo3(%arg : !fir.tdesc<!fir.type<derived7{f1:f32,f2:f32}>>)
// CHECK-LABEL: foo3
// CHECK-SAME:  !llvm.ptr<i8>

// -----

// Test nested tuple types
func.func private @foo0(%arg0: tuple<i64, tuple<f32, i64>>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.struct<(i64, struct<(f32, i64)>)>

// -----

// Test that fir.box inside tuple and derived type are lowered to struct type.
func.func private @foo0(%arg0: tuple<i64, !fir.box<i32>>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.struct<(i64, struct<(ptr<i32>, i{{.*}})>)>

func.func private @foo1(%arg0: !fir.type<derived8{a:i64,b:!fir.box<i32>}>)
// CHECK-LABEL: foo1
// CHECK-SAME: !llvm.struct<"derived8", (i64, struct<(ptr<i32>, i{{.*}})>)>

// -----

// Test fir.box<none> translation.
// `none` is used for polymorphic type.
func.func private @foo0(%arg0: !fir.box<none>)
// CHECK-LABEL: foo0
// CHECK-SAME: !llvm.ptr<struct<(ptr<struct<()>>, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}}, i{{.*}})>>)
