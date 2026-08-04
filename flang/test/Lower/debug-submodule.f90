! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-fir -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NO_DEBUG

! A submodule is not a first class entity in FIR and its name is not qualified,
! so lowering records the module at the root of its ancestry. A separate module
! procedure is mangled with the module that declares its interface, so lowering
! also records the submodule that defines it.

! NO_DEBUG-NOT: fir.module_debug_imports
! NO_DEBUG-NOT: fir.defining_submodule

! CHECK-DAG: func.func @_QMshapesPsquare({{.*}}attributes {fir.defining_submodule = "impl"}
! An ordinary module procedure needs no attribute: its own name has the
! submodule in it already.
! CHECK-DAG: func.func @_QMshapesSimplSdeepPdeep_helper() {

! CHECK-DAG: fir.module_debug_imports "shapes" {
! CHECK-DAG: fir.module_debug_imports "impl" in "shapes" {
! The ancestor of a nested submodule is the module at the root, not the
! submodule that contains it.
! CHECK-DAG: fir.module_debug_imports "deep" in "shapes" {

module shapes
  interface
    module function square(n) result(r)
      integer, intent(in) :: n
      integer :: r
    end function square
  end interface
end module shapes

submodule (shapes) impl
  integer :: sub_var = 7
contains
  module function square(n) result(r)
    integer, intent(in) :: n
    integer :: r
    r = n * n
  end function square
end submodule impl

submodule (shapes:impl) deep
  integer :: deep_var = 11
contains
  subroutine deep_helper()
    deep_var = deep_var + sub_var
  end subroutine deep_helper
end submodule deep
