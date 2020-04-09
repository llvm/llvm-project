func @r_incr(%arg0: !fir.ref<f64>) -> f64 {
    %0 = fir.alloca f64 {name = "r_incr"} : !fir.ref<f64>
    %1 = fir.load %arg0 : !fir.ref<f64>
    fir.store %1 to %0 : !fir.ref<f64>
    %2 = fir.load %arg0 : !fir.ref<f64>
    %cst = constant 1.000000e+00 : f64
    %3 = fir.addf %2, %cst : f64
    fir.store %3 to %arg0 : !fir.ref<f64>
    %4 = fir.load %0 : !fir.ref<f64>
    return %4 : f64
}

func @_MAIN() {
    %0 = fir.alloca f64 {name = "y"} : !fir.ref<f64>
    %1 = fir.alloca f64 {name = "y2"} : !fir.ref<f64>
    %2 = fir.alloca f64 {name = "y1"} : !fir.ref<f64>
    %3 = fir.alloca f64 {name = "x"} : !fir.ref<f64>
    %cst = constant 1.000000e+00 : f64
    fir.store %cst to %3 : !fir.ref<f64>
    %4 = fir.load %3 : !fir.ref<f64>
    fir.store %4 to %2 : !fir.ref<f64>
    %5 = fir.load %2 : !fir.ref<f64>
    %6 = call @r_incr(%3) : (!fir.ref<f64>) -> f64
    %7 = fir.addf %5, %6 : f64
    fir.store %7 to %1 : !fir.ref<f64>
    %8 = fir.load %1 : !fir.ref<f64>
    %9 = fir.load %3 : !fir.ref<f64>
    %10 = fir.addf %8, %9 : f64
    fir.store %10 to %0 : !fir.ref<f64>
    %c1_i32 = constant 1 : i32
    %11 = call @__F18IOa_BeginExternalListOutput(%c1_i32) : (i32) -> !fir.ref<i32>
    %12 = fir.load %0 : !fir.ref<f64>
    call @__F18IOa_OutputReal64(%11, %12) : (!fir.ref<i32>, f64) -> ()
    call @__F18IOa_EndIOStatement(%11) : (!fir.ref<i32>) -> ()
    return
}

func @__F18IOa_BeginExternalListOutput(i32) -> !fir.ref<i32>
func @__F18IOa_OutputInteger64(!fir.ref<i32>, i64)
func @__F18IOa_OutputReal64(!fir.ref<i32>, f64)
func @__F18IOa_EndIOStatement(!fir.ref<i32>)
