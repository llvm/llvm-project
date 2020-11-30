// RUN: tco --target=i386-unknown-linux-gnu %s | FileCheck %s --check-prefix=I32
// RUN: tco --target=x86_64-unknown-linux-gnu %s | FileCheck %s --check-prefix=X64

// I32-LABEL: define i64 @gen4()
// X64-LABEL: define <2 x float> @gen4()
func @gen4() -> !fir.complex<4> {
  %1 = fir.undefined !fir.complex<4>
  %2 = constant 2.0 : f32
  %3 = fir.convert %2 : (f32) -> !fir.real<4>
  %c0 = constant 0 : i32
  %4 = fir.insert_value %1, %3, %c0 : (!fir.complex<4>, !fir.real<4>, i32) -> !fir.complex<4>
  %c1 = constant 1 : i32
  %5 = constant -42.0 : f32
  %6 = fir.insert_value %4, %5, %c1 : (!fir.complex<4>, f32, i32) -> !fir.complex<4>
  // I32: store { float, float } { float 2.000000e+00, float -4.200000e+01 }
  // I32: %[[load:.*]] = load i64, i64*
  // I32: ret i64 %[[load]]
  // X64: store { float, float } { float 2.000000e+00, float -4.200000e+01 }
  // X64: %[[load:.*]] = load <2 x float>, <2 x float>*
  // X64: ret <2 x float> %[[load]]
  return %6 : !fir.complex<4>
}

// I32-LABEL: define void @gen8({ double, double }* sret %
// X64-LABEL: define { double, double } @gen8()
func @gen8() -> !fir.complex<8> {
  %1 = fir.undefined !fir.complex<8>
  %2 = constant 1.0 : f64
  %3 = constant -4.0 : f64
  %c0 = constant 0 : i32
  %4 = fir.insert_value %1, %3, %c0 : (!fir.complex<8>, f64, i32) -> !fir.complex<8>
  %c1 = constant 1 : i32
  %5 = fir.insert_value %4, %2, %c1 : (!fir.complex<8>, f64, i32) -> !fir.complex<8>
  // I32: store { double, double } { double -4.000000e+00, double 1.000000e+00 }
  // I64: store { double, double } { double -4.000000e+00, double 1.000000e+00 }
  // I64: %[[load:.*]] = load { double, double }
  // I64: ret { double, double } %[[load]]
  return %5 : !fir.complex<8>
}

// I32: declare void @sink4({ float, float }*)
// X64: declare void @sink4(<2 x float>)
func private @sink4(!fir.complex<4>) -> ()

// I32: declare void @sink8({ double, double }*)
// X64: declare void @sink8(double, double)
func private @sink8(!fir.complex<8>) -> ()

// I32-LABEL: define void @call4()
// X64-LABEL: define void @call4()
func @call4() {
  // I32: = call i64 @gen4()
  // X64: = call <2 x float> @gen4()
  %1 = fir.call @gen4() : () -> !fir.complex<4>
  // I32: call void @sink4({ float, float }* %
  // X64: call void @sink4(<2 x float> %
  fir.call @sink4(%1) : (!fir.complex<4>) -> ()
  return
}

// I32-LABEL: define void @call8()
// X64-LABEL: define void @call8()
func @call8() {
  // I32: call void @gen8({ double, double }* %
  // X64: = call { double, double } @gen8()
  %1 = fir.call @gen8() : () -> !fir.complex<8>
  // I32: call void @sink8({ double, double }* %
  // X64: call void @sink8(double %4, double %5)
  fir.call @sink8(%1) : (!fir.complex<8>) -> ()
  return
}

// I32-LABEL: define i64 @char1lensum(i8* %0, i8* %1, i32 %2, i32 %3)
// X64-LABEL: define i64 @char1lensum(i8* %0, i8* %1, i64 %2, i64 %3)
func @char1lensum(%arg0 : !fir.boxchar<1>, %arg1 : !fir.boxchar<1>) -> i64 {
  // X64-DAG: %[[p0:.*]] = insertvalue { i8*, i64 } undef, i8* %1, 0
  // X64-DAG: = insertvalue { i8*, i64 } %[[p0]], i64 %3, 1
  // X64-DAG: %[[p1:.*]] = insertvalue { i8*, i64 } undef, i8* %0, 0
  // X64-DAG: = insertvalue { i8*, i64 } %[[p1]], i64 %2, 1
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i64)
  // I32: %[[add:.*]] = add i64 %
  // X64: %[[add:.*]] = add i64 %
  %3 = addi %1#1, %2#1 : i64
  // I32: ret i64 %[[add]]
  // X64: ret i64 %[[add]]
  return %3 : i64
}

// I32-LABEL: define void @char1copy(i8* sret %0, i32 %1, i8* %2, i32 %3)
// I64-LABEL: define void @char1copy(i8* sret %0, i64 %1, i8* %2, i64 %3)
func @char1copy(%arg0 : !fir.boxchar<1> {llvm.sret = true}, %arg1 : !fir.boxchar<1>) {
  // I32-DAG: %[[p0:.*]] = insertvalue { i8*, i32 } undef, i8* %2, 0
  // I32-DAG: = insertvalue { i8*, i32 } %[[p0]], i32 %3, 1
  // I32-DAG: %[[p1:.*]] = insertvalue { i8*, i32 } undef, i8* %0, 0
  // I32-DAG: = insertvalue { i8*, i32 } %[[p1]], i32 %1, 1
  // X64-DAG: %[[p0:.*]] = insertvalue { i8*, i64 } undef, i8* %2, 0
  // X64-DAG: = insertvalue { i8*, i64 } %[[p0]], i64 %3, 1
  // X64-DAG: %[[p1:.*]] = insertvalue { i8*, i64 } undef, i8* %0, 0
  // X64-DAG: = insertvalue { i8*, i64 } %[[p1]], i64 %1, 1
  %1:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  %2:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.array<?x!fir.char<1>>>, i64)
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %3 = fir.convert %1#1 : (i64) -> index
  %last = subi %3, %c1 : index
  fir.do_loop %i = %c0 to %last step %c1 {
    %in_pos = fir.coordinate_of %2#0, %i : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    %out_pos = fir.coordinate_of %1#0, %i : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    %ch = fir.load %in_pos : !fir.ref<!fir.char<1>>
    fir.store %ch to %out_pos : !fir.ref<!fir.char<1>>
  }
  // I32: ret void
  // X64: ret void
  return
}

