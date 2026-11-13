! Check that the DO variable of a loop lowered to an acc.loop without a loop
! directive holds its final value after the loop when that value may be used,
! and that an acc.loop whose iterations run in parallel has its own copy.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

function used_after_in_routine(n) result(last)
  !$acc routine seq
  integer :: n, k, last
  do k = 1, n
  end do
  last = k
end function

! CHECK-LABEL: func.func @_QPused_after_in_routine(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_in_routineEk")
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) = (%[[LB:.*]] : i32) to (%[[UB:.*]] : i32) step (%[[STEP:.*]] : i32) {
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LB_IDX:.*]] = fir.convert %[[LB]] : (i32) -> index
! CHECK: %[[UB_IDX:.*]] = fir.convert %[[UB]] : (i32) -> index
! CHECK: %[[STEP_IDX:.*]] = fir.convert %[[STEP]] : (i32) -> index
! CHECK: %[[ZERO:.*]] = arith.constant 0 : index
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB_IDX]], %[[LB_IDX]] : index
! CHECK: %[[SPAN:.*]] = arith.addi %[[DIFF]], %[[STEP_IDX]] : index
! CHECK: %[[QUOT:.*]] = arith.divsi %[[SPAN]], %[[STEP_IDX]] : index
! CHECK: %[[POSITIVE:.*]] = arith.cmpi sgt, %[[QUOT]], %[[ZERO]] : index
! CHECK: %[[COUNT:.*]] = arith.select %[[POSITIVE]], %[[QUOT]], %[[ZERO]] : index
! CHECK: %[[STRIDE:.*]] = arith.muli %[[COUNT]], %[[STEP_IDX]] : index
! CHECK: %[[LAST:.*]] = arith.addi %[[LB_IDX]], %[[STRIDE]] : index
! CHECK: %[[LAST_I32:.*]] = fir.convert %[[LAST]] : (index) -> i32
! CHECK: fir.store %[[LAST_I32]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>

! The bounds do not have the kind of the variable.
function mixed_kinds(n) result(last)
  !$acc routine seq
  integer :: n
  integer(8) :: k, last
  do k = 1, n
  end do
  last = k
end function

! CHECK-LABEL: func.func @_QPmixed_kinds(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFmixed_kindsEk")
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i64>) control(%{{.*}} : i64) =
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LB_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[UB_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[STEP_IDX:.*]] = fir.convert %{{.*}} : (i64) -> index
! CHECK: %[[DIFF:.*]] = arith.subi %[[UB_IDX]], %[[LB_IDX]] : index
! CHECK: %[[STRIDE:.*]] = arith.muli %{{.*}}, %[[STEP_IDX]] : index
! CHECK: %[[LAST:.*]] = arith.addi %[[LB_IDX]], %[[STRIDE]] : index
! CHECK: %[[LAST_I64:.*]] = fir.convert %[[LAST]] : (index) -> i64
! CHECK: fir.store %[[LAST_I64]] to %[[K]]#0 : !fir.ref<i64>

subroutine used_after_in_serial(n, r)
  integer :: n, k, r
  !$acc serial copyout(r)
  do k = 1, n
  end do
  r = k
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPused_after_in_serial(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_in_serialEk")
! CHECK: acc.serial
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>

! The assignment is in the body of the enclosing loop, after the inner loop.
! The iterations of the enclosing loop run in parallel: each has its own copy.
subroutine used_after_inner_loop(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel loop
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_after_inner_loop(
! CHECK: %[[K_HOST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_after_inner_loopEk")
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%[[K_HOST]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFused_after_inner_loopEk")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: hlfir.assign %[[K_AFTER]] to %{{.*}} : i32, !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The variable is private to the enclosing loop: its copy is assigned.
subroutine used_in_private_loop(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel loop private(k)
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_in_private_loop(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) name("k")
! CHECK: acc.loop combined(parallel) private(%[[K_PRIVATE]], %{{[0-9]+}} : !fir.ref<i32>, !fir.ref<i32>) control(
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFused_in_private_loopEk")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>

! The copy is made whatever data clause is visible.
subroutine in_data_clause(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc data copyin(k)
  !$acc parallel loop
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end data
end subroutine

! CHECK-LABEL: func.func @_QPin_data_clause(
! CHECK: acc.data
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFin_data_clauseEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>

module do_variable_m
  integer :: km
  !$acc declare create(km)
end module

subroutine in_declare(a, kout, n)
  use do_variable_m
  integer :: n, a(n), kout(n), i
  !$acc parallel loop
  do i = 1, n
    do km = 1, a(i)
    end do
    kout(i) = km
  end do
end subroutine

! CHECK-LABEL: func.func @_QPin_declare(
! CHECK: %[[KM_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("km")
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %[[KM_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[KM:.*]]:2 = hlfir.declare %[[KM_PRIVATE]] uniq_name("_QMdo_variable_mEkm")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[KM]]#0 : !fir.ref<i32>

subroutine independent_loop_in_kernels(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc kernels
  !$acc loop independent
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPindependent_loop_in_kernels(
! CHECK: acc.kernels
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFindependent_loop_in_kernelsEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The loops of a DO CONCURRENT in a parallel construct run in parallel.
subroutine in_do_concurrent(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel
  do concurrent (i = 1:n)
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPin_do_concurrent(
! CHECK: acc.parallel
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFin_do_concurrentEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The loops that collapse associates with the acc.loop share its copy.
subroutine collapsed_loops(a, kout, n)
  integer :: n, a(n, n), kout(n, n), i, j, k
  !$acc parallel loop collapse(2)
  do i = 1, n
    do j = 1, n
      do k = 1, a(j, i)
      end do
      kout(j, i) = k
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcollapsed_loops(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFcollapsed_loopsEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: collapse([2])

! An orphaned loop in a vector routine runs its iterations in parallel.
subroutine orphaned_loop(a, kout, n)
  !$acc routine vector
  integer :: n, a(n), kout(n), i, k
  !$acc loop vector
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPorphaned_loop(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop vector private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QForphaned_loopEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The caller reads a dummy argument: it has no copy.
subroutine orphaned_loop_dummy(a, kout, n, k)
  !$acc routine vector
  integer :: n, a(n), kout(n), i, k
  !$acc loop vector
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPorphaned_loop_dummy(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QForphaned_loop_dummyEk")
! CHECK: acc.loop vector private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>

! An alias reads the variable: it has no copy.
subroutine equivalenced(a, kout, n)
  integer :: n, a(n), kout(n), i, k, j
  equivalence (k, j)
  !$acc parallel loop
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = j
  end do
end subroutine

! CHECK-LABEL: func.func @_QPequivalenced(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFequivalencedEk")
! CHECK: acc.loop combined(parallel) private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ptr<i32>

! The iterations of an auto or seq loop do not get a copy: the variable is
! assigned.
subroutine auto_loop_in_kernels(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc kernels
  !$acc loop
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPauto_loop_in_kernels(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFauto_loop_in_kernelsEk")
! CHECK: acc.kernels
! CHECK-NOT: acc.private{{.*}} name("k")
! CHECK: acc.loop private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_

subroutine seq_loop(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc parallel
  !$acc loop seq
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPseq_loop(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFseq_loopEk")
! CHECK: acc.parallel
! CHECK-NOT: acc.private{{.*}} name("k")
! CHECK: acc.loop private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) seq

! The loop is sequential for every device type.
subroutine device_type_seq(a, n, r)
  integer :: n, a(n), i, k, r
  !$acc parallel num_gangs(1) copyout(r)
  !$acc loop device_type(*) seq
  do i = 1, n
    do k = 1, a(i)
    end do
  end do
  r = k
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPdevice_type_seq(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFdevice_type_seqEk")
! CHECK: acc.parallel
! CHECK-NOT: acc.private{{.*}} name("k")
! CHECK: acc.loop private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) seq ([#acc.device_type<star>])

! The loop is independent for a device type.
subroutine device_type_independent(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc kernels
  !$acc loop device_type(nvidia) independent
  do i = 1, n
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPdevice_type_independent(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFdevice_type_independentEk")
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent ([#acc.device_type<nvidia>])

! A DO CONCURRENT with a loop directive in a kernels construct is independent.
subroutine kernels_loop_do_concurrent(a, kout, n)
  integer :: n, a(n), kout(n), i, k
  !$acc kernels loop
  do concurrent (i = 1:n)
    do k = 1, a(i)
    end do
    kout(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPkernels_loop_do_concurrent(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop combined(kernels) private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFkernels_loop_do_concurrentEk")
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! A loop lowered as control flow assigns the variable itself: it has a copy
! even when nothing reads its value after the loop.
subroutine unstructured_inner_loop(b, n)
  integer :: n, i, k
  real :: b(n, n)
  !$acc parallel loop
  do i = 1, n
    do k = 1, n
      if (b(k, i) < 0.) exit
      b(k, i) = 1.
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunstructured_inner_loop(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFunstructured_inner_loopEk")
! CHECK: cf.cond_br
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) independent

! The loop of a loop directive has its own copy of its variable, and nothing
! assigns the variable after it: the enclosing loop does not get a copy.
subroutine directed_inner_loop(a, b, n, m)
  integer :: n, m, a(m, n), b(n), i, k
  k = 42
  !$acc parallel loop gang copyin(k)
  do i = 1, n
    !$acc loop vector
    do k = 1, m
      a(k, i) = 1
    end do
    b(i) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPdirected_inner_loop(
! CHECK: %[[K_COPYIN:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<i32>) name("k")
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_COPYIN]] uniq_name("_QFdirected_inner_loopEk")
! CHECK: acc.loop combined(parallel) gang private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: acc.loop vector private(
! CHECK: } inclusiveUpperbound(array<i1: true>) independent
! CHECK-NEXT: %[[K_AFTER:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>

! The DO variable is a LOCAL_INIT variable of the DO CONCURRENT, with its own
! storage: the enclosing loop does not get a copy of the outer variable.
subroutine local_init_in_do_concurrent(a, n, m)
  integer :: n, m, a(m, n), i, j, k
  k = 7
  !$acc parallel loop
  do i = 1, n
    do concurrent (j = 1:m) local_init(k)
      a(j, i) = k
      do k = 1, 2
      end do
      a(j, i) = a(j, i) + k
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPlocal_init_in_do_concurrent(
! CHECK: %[[K_HOST:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFlocal_init_in_do_concurrentEk")
! CHECK: acc.loop combined(parallel) private(%{{[0-9]+}} : !fir.ref<i32>) control(
! CHECK: acc.firstprivate varPtr(%[[K_HOST]]#0 : !fir.ref<i32>) recipe(@firstprivatization_ref_i32) name("k")

! A local variable of a BLOCK in a routine has a copy too.
subroutine block_in_routine(a, kout, n)
  !$acc routine vector
  integer :: n, a(n), kout(n), i
  block
    integer :: k
    !$acc loop vector
    do i = 1, n
      do k = 1, a(i)
      end do
      kout(i) = k
    end do
  end block
end subroutine

! CHECK-LABEL: func.func @_QPblock_in_routine(
! CHECK: %[[K_PRIVATE:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) implicit(true) name("k")
! CHECK: acc.loop vector private(%{{.*}}, %[[K_PRIVATE]] : !fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[K:.*]]:2 = hlfir.declare %[[K_PRIVATE]] uniq_name("_QFblock_in_routineB1Ek")
! CHECK: fir.store %{{.*}} to %[[K]]#0 : !fir.ref<i32>

! A statement function reads the variable where it is referenced.
subroutine statement_function(n, r)
  integer :: n, k, r, sf, x
  sf(x) = x + k
  !$acc serial copyout(r)
  do k = 1, n
  end do
  r = sf(1)
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPstatement_function(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFstatement_functionEk")
! CHECK: acc.serial
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>

! The bounds of a later loop on the variable use its value.
subroutine used_by_next_loop(n, m, s)
  !$acc routine seq
  integer :: n, m, s, k
  do k = 1, n
    s = s + 1
  end do
  do k = k, m
    s = s + 100
  end do
end subroutine

! CHECK-LABEL: func.func @_QPused_by_next_loop(
! CHECK: %[[K:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFused_by_next_loopEk")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[K]]#0 : !fir.ref<i32>
! CHECK: %[[LB:.*]] = fir.load %[[K]]#0 : !fir.ref<i32>
! CHECK: acc.loop private(%{{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) = (%[[LB]] : i32)

! The caller reads the function result.
function result_variable(n)
  !$acc routine seq
  integer :: n, result_variable
  do result_variable = 1, n
  end do
end function

! CHECK-LABEL: func.func @_QPresult_variable(
! CHECK: %[[RES:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFresult_variableEresult_variable")
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[LAST:.*]] = fir.convert %{{.*}} : (index) -> i32
! CHECK: fir.store %[[LAST]] to %[[RES]]#0 : !fir.ref<i32>
! CHECK: %[[RES_AFTER:.*]] = fir.load %[[RES]]#0 : !fir.ref<i32>
! CHECK: return %[[RES_AFTER]] : i32

! The variable is only used in the loop: nothing is assigned after it.
subroutine not_used_after(a, n)
  !$acc routine seq
  integer :: n, k
  real :: a(n)
  do k = 1, n
    a(k) = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPnot_used_after(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: return

! Another loop on the variable and a private clause do not use its value.
subroutine only_other_loops(a, b, n)
  integer :: n, k
  real :: a(n), b(n)
  !$acc parallel private(k)
  do k = 1, n
    a(k) = 0.
  end do
  do k = 1, n
    b(k) = a(k)
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPonly_other_loops(
! CHECK: acc.parallel private(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: return

! An I/O implied DO on the variable gives it new values.
subroutine only_io_implied_do(a, n)
  !$acc routine seq
  integer :: n, k
  real :: a(n)
  do k = 1, n
    a(k) = k
  end do
  print *, (a(k), k = 1, n)
end subroutine

! CHECK-LABEL: func.func @_QPonly_io_implied_do(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: fir.call @_FortranAioBeginExternalListOutput

! The variable is private to the thread that executes the loop in the compute
! construct: a use after the construct does not see its value.
subroutine used_after_construct(a, n, r)
  integer :: n, k, r
  real :: a(n)
  !$acc kernels
  do k = 1, n
    a(k) = 0.
  end do
  !$acc end kernels
  r = k
end subroutine

! CHECK-LABEL: func.func @_QPused_after_construct(
! CHECK: acc.kernels
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) auto_
! CHECK-NOT: fir.store
! CHECK: return

! A loop with a loop directive is not changed.
subroutine explicit_loop_directive(n, r)
  integer :: n, k, r
  !$acc serial copyout(r)
  !$acc loop seq
  do k = 1, n
  end do
  r = k
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPexplicit_loop_directive(
! CHECK: acc.loop private(
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK-NOT: fir.store
! CHECK: return
