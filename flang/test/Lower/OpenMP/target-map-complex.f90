!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Check that the complex*4 is passed by value. but complex*8 is passed by
! reference

!CHECK-LABEL: func.func @_QMmPbar()
!CHECK:  %[[V0:[0-9]+]]:2 = hlfir.declare {{.*}} (!fir.ref<complex<f64>>) -> (!fir.ref<complex<f64>>, !fir.ref<complex<f64>>)
!CHECK:  %[[V1:[0-9]+]]:2 = hlfir.declare {{.*}} (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!CHECK:  %[[V2:[0-9]+]] = omp.map.info var_ptr(%[[V1]]#1 : !fir.ref<complex<f32>>, complex<f32>) {{.*}} capture(ByCopy)
!CHECK:  %[[V3:[0-9]+]] = omp.map.info var_ptr(%[[V0]]#1 : !fir.ref<complex<f64>>, complex<f64>) {{.*}} capture(ByRef)
!CHECK:  omp.target map_entries(%[[V2]] -> {{.*}}, %[[V3]] -> {{.*}} : !fir.ref<complex<f32>>, !fir.ref<complex<f64>>)

module m
  implicit none
  complex(kind=4) :: cfval = (24, 25)
  complex(kind=8) :: cdval = (28, 29)
  interface
    subroutine foo(x, y)
      complex(kind=4) :: x
      complex(kind=8) :: y
      !$omp declare target
    end
  end interface

contains

subroutine bar()
!$omp target
    call foo(cfval, cdval)
!$omp end target
end

end module
