! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: cpu_time_test
subroutine cpu_time_test(t)
    real :: t
    ! CHECK: %[[result64:[0-9]+]] = fir.call @_FortranACpuTime() {{.*}}: () -> f64
    ! CHECK: %[[result32:[0-9]+]] = fir.convert %[[result64]] : (f64) -> f32
    ! CHECK: fir.store %[[result32]] to %arg0 : !fir.ref<f32>
    call cpu_time(t)
  end subroutine
  
