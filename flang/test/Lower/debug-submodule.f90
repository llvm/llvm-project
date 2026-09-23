! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s --check-prefix=NO_DEBUG
! RUN: %flang_fc1 -emit-fir -debug-info-kind=line-tables-only %s -o - | FileCheck %s --check-prefix=NO_DEBUG

! Test that lowering records the ancestry of a submodule, and the submodule
! that defines a separate module procedure, only when debug info asks for it.

! NO_DEBUG-NOT: fir.module_debug_imports
! NO_DEBUG-NOT: fir.defining_submodule

! Only a separate module procedure needs the attribute. Any other procedure
! has the submodule in its own name already.
! CHECK-DAG: func.func @_QMshapesPsquare({{.*}}attributes {fir.defining_submodule = "impl"}
! CHECK-DAG: func.func @_QMshapesSimplSdeepPdeep_helper() {

! The parent of a first level submodule is the module at the root, and that of
! a nested one is the submodule containing it.
! CHECK-DAG: fir.module_debug_imports "shapes" {
! CHECK-DAG: fir.module_debug_imports "impl" in "shapes" parent "shapes" {
! CHECK-DAG: fir.module_debug_imports "deep" in "shapes" parent "impl" {

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
