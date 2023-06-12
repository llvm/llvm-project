! Test lowering of intrinsic procedure to HLFIR with assumed types
! arguments. These are a bit special because semantics do not represent
! assumed types actual arguments with an evaluate::Expr like for usual
! arguments.
! RUN: bbc -emit-hlfir --polymorphic-type -o - %s | FileCheck %s

subroutine assumed_type_to_intrinsic(a)
  type(*) :: a(:)
  if (is_contiguous(a)) call something()
end subroutine
! CHECK-LABEL:   func.func @_QPassumed_type_to_intrinsic(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}a"
! CHECK:  %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#1 : (!fir.box<!fir.array<?xnone>>) -> !fir.box<none>
! CHECK:  fir.call @_FortranAIsContiguous(%[[VAL_2]]) {{.*}}: (!fir.box<none>) -> i1

subroutine assumed_type_optional_to_intrinsic(a)
  type(*), optional :: a(:)
  if (present(a)) call something()
end subroutine
! CHECK-LABEL:   func.func @_QPassumed_type_optional_to_intrinsic(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}a"
! CHECK:  fir.is_present %[[VAL_1]]#1 : (!fir.box<!fir.array<?xnone>>) -> i1
