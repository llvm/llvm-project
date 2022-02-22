! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

integer(1) function fct1()
end
! CHECK-LABEL: func @_QPfct1() -> i8
! CHECK:         return %{{.*}} : i8

integer(2) function fct2()
end
! CHECK-LABEL: func @_QPfct2() -> i16
! CHECK:         return %{{.*}} : i16

integer(4) function fct3()
end
! CHECK-LABEL: func @_QPfct3() -> i32
! CHECK:         return %{{.*}} : i32

integer(8) function fct4()
end
! CHECK-LABEL: func @_QPfct4() -> i64
! CHECK:         return %{{.*}} : i64

integer(16) function fct5()
end
! CHECK-LABEL: func @_QPfct5() -> i128
! CHECK:         return %{{.*}} : i128

function fct()
  integer :: fct
end
! CHECK-LABEL: func @_QPfct() -> i32
! CHECK:         return %{{.*}} : i32

function fct_res() result(res)
  integer :: res
end
! CHECK-LABEL: func @_QPfct_res() -> i32
! CHECK:         return %{{.*}} : i32

integer function fct_body()
  goto 1
  1 stop
end

! CHECK-LABEL: func @_QPfct_body() -> i32
! CHECK:         cf.br ^bb1
! CHECK:       ^bb1
! CHECK:         %{{.*}} = fir.call @_FortranAStopStatement
! CHECK:         fir.unreachable

function fct_iarr1()
  integer, dimension(10) :: fct_iarr1
end

! CHECK-LABEL: func @_QPfct_iarr1() -> !fir.array<10xi32>
! CHECK:         return %{{.*}} : !fir.array<10xi32>

function fct_iarr2()
  integer, dimension(10, 20) :: fct_iarr2
end

! CHECK-LABEL: func @_QPfct_iarr2() -> !fir.array<10x20xi32>
! CHECK:         return %{{.*}} : !fir.array<10x20xi32>

function fct_iarr3()
  integer, dimension(:, :), allocatable :: fct_iarr3
end

! CHECK-LABEL: func @_QPfct_iarr3() -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
! CHECK:        return %{{.*}} : !fir.box<!fir.heap<!fir.array<?x?xi32>>>

function fct_iarr4()
  integer, dimension(:), pointer :: fct_iarr4
end

! CHECK-LABEL: func @_QPfct_iarr4() -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         return %{{.*}} : !fir.box<!fir.ptr<!fir.array<?xi32>>>

logical(1) function lfct1()
end
! CHECK-LABEL: func @_QPlfct1() -> !fir.logical<1>
! CHECK:         return %{{.*}} : !fir.logical<1>

logical(2) function lfct2()
end
! CHECK-LABEL: func @_QPlfct2() -> !fir.logical<2>
! CHECK:         return %{{.*}} : !fir.logical<2>

logical(4) function lfct3()
end
! CHECK-LABEL: func @_QPlfct3() -> !fir.logical<4>
! CHECK:         return %{{.*}} : !fir.logical<4>

logical(8) function lfct4()
end
! CHECK-LABEL: func @_QPlfct4() -> !fir.logical<8>
! CHECK:         return %{{.*}} : !fir.logical<8>

real(2) function rfct1()
end
! CHECK-LABEL: func @_QPrfct1() -> f16
! CHECK:         return %{{.*}} : f16

real(3) function rfct2()
end
! CHECK-LABEL: func @_QPrfct2() -> bf16
! CHECK:         return %{{.*}} : bf16

real function rfct3()
end
! CHECK-LABEL: func @_QPrfct3() -> f32
! CHECK:         return %{{.*}} : f32

real(8) function rfct4()
end
! CHECK-LABEL: func @_QPrfct4() -> f64
! CHECK:         return %{{.*}} : f64

real(10) function rfct5()
end
! CHECK-LABEL: func @_QPrfct5() -> f80
! CHECK:         return %{{.*}} : f80

real(16) function rfct6()
end
! CHECK-LABEL: func @_QPrfct6() -> f128
! CHECK:         return %{{.*}} : f128

complex(2) function cplxfct1()
end
! CHECK-LABEL: func @_QPcplxfct1() -> !fir.complex<2>
! CHECK:         return %{{.*}} : !fir.complex<2>

complex(3) function cplxfct2()
end
! CHECK-LABEL: func @_QPcplxfct2() -> !fir.complex<3>
! CHECK:         return %{{.*}} : !fir.complex<3>

complex(4) function cplxfct3()
end
! CHECK-LABEL: func @_QPcplxfct3() -> !fir.complex<4>
! CHECK:         return %{{.*}} : !fir.complex<4>

complex(8) function cplxfct4()
end
! CHECK-LABEL: func @_QPcplxfct4() -> !fir.complex<8>
! CHECK:         return %{{.*}} : !fir.complex<8>

complex(10) function cplxfct5()
end
! CHECK-LABEL: func @_QPcplxfct5() -> !fir.complex<10>
! CHECK:         return %{{.*}} : !fir.complex<10>

complex(16) function cplxfct6()
end
! CHECK-LABEL: func @_QPcplxfct6() -> !fir.complex<16>
! CHECK:         return %{{.*}} : !fir.complex<16>
