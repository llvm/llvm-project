! Test that lowering makes a difference between NAME="" and no NAME
! in BIND(C). See Fortran 2018 standard 18.10.2 point 2.
! BIND(C, NAME="") implies there is no binding label, meaning that
! the Fortran mangled name has to be used.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

!CHECK: func.func @_QPfoo(%{{.*}}: !fir.ref<i16>
subroutine foo(x) bind(c, name="")
  integer(2) :: x
end subroutine

!CHECK: func.func @bar(%{{.*}}: !fir.ref<i32>
subroutine foo(x) bind(c, name="bar")
  integer(4) :: x
end subroutine

!CHECK: func.func @_QMinamodule1Pfoo(%{{.*}}: !fir.ref<i64>
module inamodule1
contains
subroutine foo(x) bind(c, name="")
  integer(8) :: x
end subroutine
end module
