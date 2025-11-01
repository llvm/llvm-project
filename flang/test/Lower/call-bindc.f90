! Chekc that BIND(C) is carried over to the fir.call
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module mod
  interface
    complex(kind=4) function foo4c(j) bind(c)
      integer, intent(in),value :: j
    end function foo4c
  end interface
end module mod

program main
  use mod
  procedure(foo4c), pointer :: fptr4c
  complex(kind=4) :: res4
  fptr4c => foo4c
  res4 = fptr4c(6)
end

! CHECK-LABEL: func.func @_QQmain()
! CHECK: fir.call %{{.*}}(%{{.*}}) proc_attrs<bind_c> fastmath<contract> : (i32) -> complex<f32>
