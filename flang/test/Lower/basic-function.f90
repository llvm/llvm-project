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
