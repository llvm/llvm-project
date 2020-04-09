func @foo() -> f32 {
    %0 = fir.alloca f32 {name = "foo"} : !fir.ref<f32>
    %1 = fir.alloca f32 {name = "x"} : !fir.ref<f32>
    %cst = constant 4.200000e+01 : f32
    fir.store %cst to %1 : !fir.ref<f32>
    %cst_0 = constant 6.600000e+01 : f32
    fir.store %cst_0 to %1 : !fir.ref<f32>
    %2 = fir.load %1 : !fir.ref<f32>
    fir.store %2 to %0 : !fir.ref<f32>
    %3 = fir.load %0 : !fir.ref<f32>
    return %3 : f32
}
