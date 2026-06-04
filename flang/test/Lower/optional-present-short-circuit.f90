! Test short-circuit evaluation of a scalar .AND./.OR. whose left operand tests
! the presence of an OPTIONAL argument. The right operand -- which reads the
! descriptor of the optional argument -- must be lowered inside an fir.if so it
! is not evaluated (and the descriptor not dereferenced) when the argument is
! absent. Fortran does not require short-circuit evaluation, but the standard
! permits not evaluating an operand whose value is not needed (F2023 10.1.7).

! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPand_guard(
! CHECK-SAME:      %[[FIELD:.*]]: !fir.box<!fir.array<?x?x?xf32>> {{.*}}"field"
! CHECK-SAME:      %[[MASK:.*]]: !fir.box<!fir.array<?x?x?xf32>> {{.*}}"mask"
subroutine and_guard(field, mask)
  real, intent(in) :: field(:,:,:)
  real, intent(in), optional :: mask(:,:,:)
  ! CHECK: %[[MASKD:.*]]:2 = hlfir.declare %[[MASK]]
  ! CHECK: %[[PRES:.*]] = fir.is_present %[[MASKD]]#1
  ! CHECK: %{{.*}} = fir.if %[[PRES]] -> (!fir.logical<4>) {
  ! The descriptor read of the optional argument is emitted only inside the
  ! presence-guarded region, with no second presence test around it.
  ! CHECK-NOT: fir.is_present
  ! CHECK: fir.box_dims %[[MASKD]]#0
  ! CHECK: } else {
  ! CHECK: arith.constant false
  ! CHECK: }
  ! CHECK-NOT: arith.andi
  if (present(mask) .and. (size(mask,3) /= size(field,3))) then
    error stop 'inconsistent dimensions'
  end if
end subroutine

! Chained presence tests: the descriptor reads of both optional arguments are
! evaluated only after all the presence tests pass.
! CHECK-LABEL: func.func @_QPchained_and_guard(
subroutine chained_and_guard(a, b)
  real, intent(in), optional :: a(:,:,:)
  real, intent(in), optional :: b(:,:,:)
  ! CHECK: %[[AD:.*]]:2 = hlfir.declare {{.*}}Ea"
  ! CHECK: %[[BD:.*]]:2 = hlfir.declare {{.*}}Eb"
  ! present(a) .and. present(b) is itself short-circuited.
  ! CHECK: %[[PA:.*]] = fir.is_present %[[AD]]#1
  ! CHECK: fir.if %[[PA]] -> (!fir.logical<4>) {
  ! CHECK: fir.is_present %[[BD]]#1
  ! CHECK: }
  ! The size comparison (descriptor reads of a and b) is nested under the
  ! combined presence test.
  ! CHECK: fir.if %{{.*}} -> (!fir.logical<4>) {
  ! CHECK: fir.box_dims %[[AD]]#0
  ! CHECK: fir.box_dims %[[BD]]#0
  if (present(a) .and. present(b) .and. (size(a,3) /= size(b,3))) then
    error stop 'mismatch'
  end if
end subroutine

! .OR. short-circuits when the left operand is true, so the right operand (the
! descriptor read) is lowered in the else region and only runs when the
! argument is present.
! CHECK-LABEL: func.func @_QPor_guard(
subroutine or_guard(mask)
  real, intent(in), optional :: mask(:,:,:)
  ! CHECK: %[[MASKD:.*]]:2 = hlfir.declare {{.*}}Emask"
  ! CHECK: %[[PRES:.*]] = fir.is_present %[[MASKD]]#1
  ! CHECK: %[[NOTPRES:.*]] = arith.xori %[[PRES]], %{{.*}} : i1
  ! CHECK: fir.if %[[NOTPRES]] -> (!fir.logical<4>) {
  ! CHECK: } else {
  ! CHECK: fir.box_dims %[[MASKD]]#0
  ! CHECK: }
  if (.not. present(mask) .or. (size(mask,3) == 1)) then
    error stop 'absent or trivial'
  end if
end subroutine
