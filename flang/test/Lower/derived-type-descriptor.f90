! Test lowering of derived type descriptors builtin data
! RUN: %bbc -emit-fir %s -o - | FileCheck %s

subroutine foo()
  real, save, target :: init_values(10, 10)
  type sometype
    integer :: num = 42
    real, pointer :: values(:, :) => init_values
  end type
  type(sometype), allocatable, save :: x(:)
end subroutine

! FIXME: n.xxx and di.xxx symbol mangling is not bijective (lacks Ffoo prefix)
! because the related symbols are inside sometype derived type scope, and FIR
! mangling was not made to support global in such scopes. Whether lowering
! should handle this or the symbols should be in foo scope is under discussion.

! CHECK-DAG: fir.global internal @_QE.n.num("num") : !fir.char<1,3>
! CHECK-DAG: fir.global internal @_QE.n.values("values") : !fir.char<1,6>
! CHECK-DAG: fir.global internal @_QE.di.sometype.num : i32
! CHECK-DAG: fir.global internal @_QFfooE.n.sometype("sometype") : !fir.char<1,8>

! CHECK-LABEL: fir.global internal @_QFfooE.c.sometype {{.*}} {
  ! CHECK: fir.address_of(@_QE.n.num)
  ! CHECK: fir.address_of(@_QE.di.sometype.num) : !fir.ref<i32>
  ! CHECK: fir.address_of(@_QE.n.values)
  ! CHECK: fir.address_of(@_QFfooEinit_values)
! CHECK: }

! CHECK-LABEL: fir.global internal @_QFfooE.dt.sometype {{.*}} {
  !CHECK: fir.address_of(@_QFfooE.n.sometype)
  !CHECK: fir.address_of(@_QFfooE.c.sometype)
! CHECK:}
