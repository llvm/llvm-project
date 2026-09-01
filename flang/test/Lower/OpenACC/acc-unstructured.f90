! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test_unstructured1(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc data copy(a, b, c)

  !$acc kernels
  a(:,:,:) = 0.0
  !$acc end kernels

  !$acc kernels
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do
  end do
  !$acc end kernels

  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
      end do
    end do

    if (a(1,2,3) > 10) stop 'just to be unstructured'
  end do

  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured1
! CHECK: acc.data
! CHECK: acc.kernels
! CHECK: acc.kernels
! CHECK: fir.call @_FortranAStopStatementText


! Body looks unstructured (if/stop) but the wrap-in-execute-region pass hides
! the unstructured CFG inside scf.execute_region, so the DOs lower as
! structured acc.loop control(...) = ... (no `unstructured` attribute). GOTO
! exiting a combined OpenACC region is not yet implemented in lowering, so
! there's no genuinely-unstructured counterpart for this combined form.
subroutine test_unstructured2(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc serial loop
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do

! CHECK-LABEL: func.func @_QPtest_unstructured2
! CHECK: acc.serial combined(loop) {
! CHECK: acc.loop combined(serial) private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText

end subroutine

subroutine test_unstructured3(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
  !$acc end parallel

! CHECK-LABEL: func.func @_QPtest_unstructured3
! CHECK: acc.parallel
! CHECK: fir.call @_FortranAStopStatementText
! CHECK: acc.yield
! CHECK: acc.yield

end subroutine

! Body looks unstructured at the source level (if/stop), but the PFT-to-MLIR
! wrap-in-execute-region pass hides the if/stop CFG inside scf.execute_region,
! so genACCDataOp sees a structured body and (with only an if-clause and no
! data clauses) hits the early-return that skips acc.data.
subroutine test_unstructured4(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) stop 'unstructured'
    end do
  end do
  !$acc end data

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured4
! CHECK-NOT: acc.data
! CHECK: fir.do_loop
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText

! Body is genuinely unstructured: GOTO exits the data region, so the
! enclosing if-construct has an external branch and is not wrappable.
! eval.lowerAsUnstructured() remains true and acc.data must be emitted.
subroutine test_unstructured4_goto(a, n)
  integer :: n, i
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    if (a(i) > 0.0) goto 100
  end do
  !$acc end data
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured4_goto
! CHECK: acc.data if(%{{.*}}) {
! CHECK: acc.terminator
! CHECK: }

! Test that GOTO exiting acc.data (one level) generates acc.terminator
! instead of an invalid cross-region branch.
subroutine test_unstructured5(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu

  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  !$acc end data
999 continue

end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured5
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br

! Test that GOTO exiting acc.loop (one level) generates acc.yield
! instead of an invalid cross-region branch.
subroutine test_unstructured6(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine seq
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
20  B(i) = A(i)
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured6
! CHECK: acc.loop gang vector
! CHECK: acc.loop
! CHECK: arith.cmpf ogt
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: } attributes {seq = [#acc.device_type<none>], unstructured}

! Test GOTO exiting acc.loop with intermediate code between loop end and
! target. A jump table (exit selector + dispatch) skips the intermediate code.
subroutine test_unstructured7(A, B, C, N)
  implicit real*8 (a-h, o-z)
  !$acc routine seq
  dimension A(*), B(*), C(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
    C(i) = 999.0
20  B(i) = A(i)
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured7
! CHECK: acc.loop gang vector
! Inner loop stores exit selector and yields:
! CHECK: acc.loop
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: } attributes {seq = [#acc.device_type<none>], unstructured}
! Jump table after inner loop:
! CHECK: fir.load %{{.*}} : !fir.ref<i32>
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br
! Intermediate code on fall-through path:
! CHECK: arith.constant 9.990000e+02

! Test GOTO exiting acc.data with intermediate code. Jump table dispatches
! after the acc.data op.
subroutine test_unstructured8(a, n)
  integer :: n, i, j
  real :: a(:)
  logical :: use_gpu
  use_gpu = .true.
  !$acc data if(use_gpu)
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  a(1) = -1.0
  !$acc end data
999 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured8
! Inside acc.data, GOTO stores exit selector and terminates:
! CHECK: acc.data if(%{{.*}}) {
! CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
! CHECK: acc.terminator
! CHECK: acc.terminator
! CHECK: }
! Jump table after acc.data:
! CHECK: fir.load %{{.*}} : !fir.ref<i32>
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br

! Test that `acc serial loop collapse(N)` whose body has an early-exit
! (here, `if (cond) then ... cycle ... end if`) lowers cleanly. The
! corresponding acc.loop must privatize all N induction variables, carry
! both `collapse = [N]` and `unstructured` attributes, and emit the
! iteration mechanics for all N levels as explicit cf inside the body.
! Reproducer derived from lorado issue #2856.
subroutine test_unstructured_collapse_cycle(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc serial loop collapse(2) copy(a)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
  !$acc end serial loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_collapse_cycle
! CHECK: acc.serial combined(loop)
! Both induction variables (j and i) are privatized:
! CHECK: %[[PRIVJ:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK: %[[PRIVI:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! No control(...) on acc.loop — bounds are not on the op:
! CHECK: acc.loop combined(serial) private(%[[PRIVJ]], %[[PRIVI]] : !fir.ref<i32>, !fir.ref<i32>) {
! Outer loop trip-count test (j) emitted as cf:
! CHECK: arith.cmpi sgt
! CHECK: cf.cond_br
! Inner loop trip-count test (i) emitted as cf:
! CHECK: arith.cmpi sgt
! CHECK: cf.cond_br
! The if/cycle is a structured cf branch in the body:
! CHECK: arith.cmpi eq
! CHECK: cf.cond_br
! CHECK: acc.yield
! CHECK: }

! `acc serial loop collapse(N)` with STOP in body: wrap-in-execute-region hides
! the unstructured if/stop and the three collapsed iterators lower as a single
! structured acc.loop control(...) (no `unstructured` attribute).
subroutine test_unstructured_collapse_stop(a)
  integer :: i, j, k
  real :: a(:,:,:)
  !$acc serial loop collapse(3)
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_collapse_stop
! All three IVs privatized:
! CHECK: acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK: acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "k"}
! CHECK: acc.loop combined(serial) private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText
! CHECK-NOT: unstructured
! CHECK: } attributes {collapse = [3]{{.*}}}

! Test orphaned `acc loop collapse(N)`
subroutine test_unstructured_collapse_loop_only(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc loop collapse(2)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_collapse_loop_only
! Standalone acc.loop (no `combined(...)`):
! CHECK: acc.loop private(%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], independent = [#acc.device_type<none>], unstructured}

! Standalone `acc loop seq` with STOP: wrap-in-execute-region hides the
! if/stop and the DO lowers as structured acc.loop control(...) (no
! `unstructured` attribute).
subroutine test_unstructured_loop_seq_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop seq
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_seq_stop
! CHECK: acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText
! CHECK-NOT: unstructured
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>]{{.*}}}

! Same loop but the if-construct has a GOTO exiting all loops, so the
! if-construct is not wrappable, the DO remains unstructured, and acc.loop
! emits the unstructured form with the `unstructured` attribute.
subroutine test_unstructured_loop_seq_goto(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop seq
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) goto 100
    end do
  end do
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_seq_goto
! CHECK: acc.loop private({{.*}}) {
! CHECK: cf.br
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

! Standalone `acc loop auto` with STOP: same wrap-makes-structured behavior.
subroutine test_unstructured_loop_auto_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop auto
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_auto_stop
! CHECK: acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText
! CHECK-NOT: unstructured
! CHECK: } attributes {auto_ = [#acc.device_type<none>]{{.*}}}

! Same loop with GOTO exit: genuinely unstructured, `unstructured` attribute
! is emitted on acc.loop.
subroutine test_unstructured_loop_auto_goto(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc loop auto
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) goto 100
    end do
  end do
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_auto_goto
! CHECK: acc.loop private({{.*}}) {
! CHECK: cf.br
! CHECK: } attributes {auto_ = [#acc.device_type<none>], {{.*}}unstructured}

! Standalone `acc loop` inside `acc serial` with STOP: wrap-makes-structured.
subroutine test_unstructured_loop_in_serial_stop(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc serial
  !$acc loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_loop_in_serial_stop
! CHECK: acc.serial
! CHECK: acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText

! Orphan `acc loop` inside a `seq` acc routine with STOP: wrap-makes-structured.
subroutine test_unstructured_orphan_loop_in_seq_routine(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc routine seq
  !$acc loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) stop 'unstructured'
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_orphan_loop_in_seq_routine
! CHECK: acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatementText

! Same orphan loop with GOTO exit: genuinely unstructured.
subroutine test_unstructured_orphan_loop_in_seq_routine_goto(a)
  integer :: i, j
  real :: a(:,:,:)
  !$acc routine seq
  !$acc loop
  do i = 1, 10
    do j = 1, 10
      if (a(1,2,3) > 10.0) goto 100
    end do
  end do
100 continue
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_orphan_loop_in_seq_routine_goto
! CHECK: acc.loop private({{.*}}) {
! CHECK: cf.br
! CHECK: } attributes {{{.*}}seq = [#acc.device_type<none>], unstructured}

! DO loop with STOP inside `!$acc kernels`. Previously flagged as
! "unstructured do loop in acc kernels" (TODO); wrap-in-execute-region now
! hides the if/stop CFG inside scf.execute_region so the DO itself lowers as
! a structured acc.loop with control bounds.
subroutine test_unstructured_kernels_do_stop()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  !$acc kernels
  do i = 1, n
    a(i) = b(i) + 1.0
    if (i == 5) stop
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_kernels_do_stop
! CHECK: acc.kernels
! CHECK: acc.loop private({{.*}}) control({{.*}}) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
! CHECK: scf.execute_region
! CHECK: fir.call @_FortranAStopStatement

! `!$acc parallel loop` (combined construct) with STOP in the innermost
! body. wrap-in-execute-region hides the if/stop CFG so the combined
! construct still lowers as structured acc.parallel + acc.loop.
subroutine test_unstructured_parallel_loop_stop(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel loop
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_parallel_loop_stop
! CHECK: acc.parallel combined(loop)
! CHECK: acc.loop combined(parallel)

! `!$acc parallel loop collapse(3)` with STOP in the innermost body. Same
! wrap behavior as above with an added collapse clause.
subroutine test_unstructured_parallel_loop_collapse3_stop(a)
  integer :: i, j, k
  real :: a(:,:,:)
  !$acc parallel loop collapse(3)
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_unstructured_parallel_loop_collapse3_stop
! CHECK: acc.parallel combined(loop)
! CHECK: acc.loop combined(parallel)
