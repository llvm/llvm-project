// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu no-complex-conversion" %s | FileCheck --check-prefix=CMPLXOFF %s
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu no-character-conversion" %s | FileCheck --check-prefix=CHAROFF %s

// Verify that selective TargetRewrite with no-complex-conversion and
// no-character-conversion options works as expected.
//
// Original code:
// complex function test(value)
//   interface
//      complex function test1(name, c)
//        character(*) :: name
//        complex, value :: c
//      end function
//      character(10) function test2(c)
//        complex, value :: c
//      end function
//   end interface
//   complex, value :: value
//   character(10) :: r
//   test = test1("", value)
//   r = test2(value)
// end function

// CMPLXOFF-DAG: fir.call @_QPtest1({{.*}}, {{.*}}, {{.*}}) : (!fir.ref<!fir.char<1,?>>, complex<f32>, i64) -> complex<f32>
// CMPLXOFF-DAG: fir.call @_QPtest2({{.*}}, {{.*}}, {{.*}}) : (!fir.ref<!fir.char<1,10>>, index, complex<f32>) -> !fir.boxchar<1>
// CMPLXOFF-DAG: func.func private @_QPtest1(!fir.ref<!fir.char<1,?>>, complex<f32>, i64) -> complex<f32>
// CMPLXOFF-DAG: func.func private @_QPtest2(!fir.ref<!fir.char<1,10>>, index, complex<f32>) -> !fir.boxchar<1>

// CHAROFF-DAG: fir.call @_QPtest1({{.*}}, {{.*}}) : (!fir.boxchar<1>, !fir.vector<2:f32>) -> !fir.vector<2:f32>
// CHAROFF-DAG: fir.call @_QPtest2({{.*}}, {{.*}}, {{.*}}) : (!fir.ref<!fir.char<1,10>>, index, !fir.vector<2:f32>) -> !fir.boxchar<1>
// CHAROFF-DAG: func.func private @_QPtest1(!fir.boxchar<1>, !fir.vector<2:f32>) -> !fir.vector<2:f32>
// CHAROFF-DAG: func.func private @_QPtest2(!fir.ref<!fir.char<1,10>>, index, !fir.vector<2:f32>) -> !fir.boxchar<1>

module {
  func.func @_QPtest(%arg0: complex<f32> {fir.bindc_name = "value"}) -> complex<f32> {
    %c10 = arith.constant 10 : index
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %0 = fir.alloca !fir.char<1,10> {bindc_name = ".result"}
    %1 = fir.alloca complex<f32>
    fir.store %arg0 to %1 : !fir.ref<complex<f32>>
    %2 = fir.alloca !fir.char<1,10> {bindc_name = "r", uniq_name = "_QFtestEr"}
    %3 = fir.alloca complex<f32> {bindc_name = "test", uniq_name = "_QFtestEtest"}
    %4 = fir.address_of(@_QQclX) : !fir.ref<!fir.char<1,0>>
    %5 = fir.convert %4 : (!fir.ref<!fir.char<1,0>>) -> !fir.ref<!fir.char<1,?>>
    %6 = fir.emboxchar %5, %c0 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
    %7 = fir.load %1 : !fir.ref<complex<f32>>
    %8 = fir.call @_QPtest1(%6, %7) fastmath<contract> : (!fir.boxchar<1>, complex<f32>) -> complex<f32>
    fir.store %8 to %3 : !fir.ref<complex<f32>>
    %9 = fir.load %1 : !fir.ref<complex<f32>>
    %10 = fir.call @llvm.stacksave() fastmath<contract> : () -> !fir.ref<i8>
    %11 = fir.call @_QPtest2(%0, %c10, %9) fastmath<contract> : (!fir.ref<!fir.char<1,10>>, index, complex<f32>) -> !fir.boxchar<1>
    %12 = fir.convert %c10 : (index) -> i64
    %13 = fir.convert %2 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
    %14 = fir.convert %0 : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
    fir.call @llvm.memmove.p0.p0.i64(%13, %14, %12, %false) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
    fir.call @llvm.stackrestore(%10) fastmath<contract> : (!fir.ref<i8>) -> ()
    %15 = fir.load %3 : !fir.ref<complex<f32>>
    return %15 : complex<f32>
  }
  func.func private @_QPtest1(!fir.boxchar<1>, complex<f32>) -> complex<f32>
  fir.global linkonce @_QQclX constant : !fir.char<1,0> {
    %0 = fir.string_lit ""(0) : !fir.char<1,0>
    fir.has_value %0 : !fir.char<1,0>
  }
  func.func private @_QPtest2(!fir.ref<!fir.char<1,10>>, index, complex<f32>) -> !fir.boxchar<1>
  func.func private @llvm.stacksave() -> !fir.ref<i8>
  func.func private @llvm.memmove.p0.p0.i64(!fir.ref<i8>, !fir.ref<i8>, i64, i1)
  func.func private @llvm.stackrestore(!fir.ref<i8>)
}
