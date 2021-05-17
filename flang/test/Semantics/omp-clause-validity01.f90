! RUN: %S/test_errors.sh %s %t %flang_fc1 -fopenmp
use omp_lib
! Check OpenMP clause validity for the following directives:
!
!    2.5 PARALLEL construct
!    2.7.1 Loop construct
!    ...

! TODO: all the internal errors

  integer :: b = 128
  integer :: z, c = 32
  integer, parameter :: num = 16
  real(8) :: arrayA(256), arrayB(512)

  integer(omp_memspace_handle_kind) :: xy_memspace = omp_default_mem_space
  type(omp_alloctrait) :: xy_traits(1) = [omp_alloctrait(omp_atk_alignment,64)]
  integer(omp_allocator_handle_kind) :: xy_alloc
  xy_alloc = omp_init_allocator(xy_memspace, 1, xy_traits)

  arrayA = 1.414
  arrayB = 3.14
  N = 1024

! 2.5 parallel-clause -> if-clause |
!                        num-threads-clause |
!                        default-clause |
!                        private-clause |
!                        firstprivate-clause |
!                        shared-clause |
!                        copyin-clause |
!                        reduction-clause |
!                        proc-bind-clause |
!                        allocate-clause

  !$omp parallel
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel private(b) allocate(b)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel private(c, b) allocate(omp_default_mem_space : b, c)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel allocate(b) allocate(c) private(b, c)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel allocate(xy_alloc :b) private(b)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp task private(b) allocate(b)
  do i = 1, N
     z = 2
  end do
  !$omp end task

  !$omp teams private(b) allocate(b)
  do i = 1, N
     z = 2
  end do
  !$omp end teams

  !$omp target private(b) allocate(b)
  do i = 1, N
     z = 2
  end do
  !$omp end target

  !ERROR: ALLOCATE clause is not allowed on the TARGET DATA directive
  !$omp target data map(from: b) allocate(b)
  do i = 1, N
     z = 2
  enddo
   !$omp end target data

  !ERROR: SCHEDULE clause is not allowed on the PARALLEL directive
  !$omp parallel schedule(static)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: COLLAPSE clause is not allowed on the PARALLEL directive
  !$omp parallel collapse(2)
  do i = 1, N
     do j = 1, N
        a = 3.14
     enddo
  enddo
  !$omp end parallel

  !ERROR: The parameter of the COLLAPSE clause must be a constant positive integer expression
  !$omp do collapse(-1)
  do i = 1, N
    do j = 1, N
      a = 3.14
    enddo
  enddo
  !$omp end do

  a = 1.0
  !$omp parallel firstprivate(a)
  do i = 1, N
     a = 3.14
  enddo
  !ERROR: NUM_THREADS clause is not allowed on the END PARALLEL directive
  !$omp end parallel num_threads(4)

  !ERROR: LASTPRIVATE clause is not allowed on the PARALLEL directive
  !ERROR: NUM_TASKS clause is not allowed on the PARALLEL directive
  !ERROR: INBRANCH clause is not allowed on the PARALLEL directive
  !$omp parallel lastprivate(a) NUM_TASKS(4) inbranch
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: At most one NUM_THREADS clause can appear on the PARALLEL directive
  !$omp parallel num_threads(2) num_threads(4)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: The parameter of the NUM_THREADS clause must be a positive integer expression
  !$omp parallel num_threads(1-4)
  do i = 1, N
     a = 3.14
  enddo
  !ERROR: NOWAIT clause is not allowed on the END PARALLEL directive
  !$omp end parallel nowait

  !$omp parallel num_threads(num-10)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel num_threads(b+1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel
  do i = 1, N
  enddo
  !ERROR: Unmatched END TARGET directive
  !$omp end target

  ! OMP 5.0 - 2.6 Restriction point 1
  outofparallel: do k =1, 10
  !$omp parallel
  !$omp do
  outer: do i=0, 10
    inner: do j=1, 10
      exit
      exit outer
      !ERROR: EXIT to construct 'outofparallel' outside of PARALLEL construct is not allowed
      !ERROR: EXIT to construct 'outofparallel' outside of DO construct is not allowed
      exit outofparallel
    end do inner
  end do outer
  !$end omp do
  !$omp end parallel
  end do outofparallel

! 2.7.1  do-clause -> private-clause |
!                     firstprivate-clause |
!                     lastprivate-clause |
!                     linear-clause |
!                     reduction-clause |
!                     schedule-clause |
!                     collapse-clause |
!                     ordered-clause

  !ERROR: When SCHEDULE clause has AUTO specified, it must not have chunk size specified
  !ERROR: At most one SCHEDULE clause can appear on the DO directive
  !ERROR: When SCHEDULE clause has RUNTIME specified, it must not have chunk size specified
  !$omp do schedule(auto, 2) schedule(runtime, 2)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: A modifier may not be specified in a LINEAR clause on the DO directive
  !$omp do linear(ref(b))
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The NONMONOTONIC modifier can only be specified with SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)
  !ERROR: The NONMONOTONIC modifier cannot be specified if an ORDERED clause is specified
  !$omp do schedule(NONMONOTONIC:static) ordered
  do i = 1, N
     a = 3.14
  enddo

  !$omp do schedule(simd, monotonic:dynamic)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: Clause LINEAR is not allowed if clause ORDERED appears on the DO directive
  !ERROR: The parameter of the ORDERED clause must be a constant positive integer expression
  !$omp do ordered(1-1) private(b) linear(b) linear(a)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the ORDERED clause must be greater than or equal to the parameter of the COLLAPSE clause
  !$omp do collapse(num-14) ordered(1)
  do i = 1, N
     do j = 1, N
        do k = 1, N
           a = 3.14
        enddo
     enddo
  enddo

  !$omp parallel do simd if(parallel:a>1.)
  do i = 1, N
  enddo
  !$omp end parallel do simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel do if(target:a>1.)
  do i = 1, N
  enddo
  !ERROR: Unmatched END SIMD directive
  !$omp end simd

! 2.7.2 sections-clause -> private-clause |
!                         firstprivate-clause |
!                         lastprivate-clause |
!                         reduction-clause

  !$omp parallel
  !$omp sections
  !$omp section
  a = 0.0
  !$omp section
  b = 1
  !$omp end sections nowait
  !$omp end parallel

  !$omp parallel
  !$omp sections
  !$omp section
  a = 0.0
  !ERROR: Unmatched END PARALLEL SECTIONS directive
  !$omp end parallel sections
  !$omp end parallel

  !$omp parallel
  !$omp sections
  a = 0.0
  b = 1
  !$omp section
  c = 1
  d = 2
  !ERROR: NUM_THREADS clause is not allowed on the END SECTIONS directive
  !$omp end sections num_threads(4)

  !$omp parallel
  !$omp sections
    b = 1
  !$omp section
    c = 1
    d = 2
  !ERROR: At most one NOWAIT clause can appear on the END SECTIONS directive
  !$omp end sections nowait nowait
  !$omp end parallel

  !$omp end parallel

! 2.11.2 parallel-sections-clause -> parallel-clause |
!                                    sections-clause

  !$omp parallel sections num_threads(4) private(b) lastprivate(d)
  a = 0.0
  !$omp section
  b = 1
  c = 2
  !$omp section
  d = 3
  !$omp end parallel sections

  !ERROR: At most one NUM_THREADS clause can appear on the PARALLEL SECTIONS directive
  !$omp parallel sections num_threads(1) num_threads(4)
  a = 0.0
  !ERROR: Unmatched END SECTIONS directive
  !$omp end sections

  !$omp parallel sections
  !ERROR: NOWAIT clause is not allowed on the END PARALLEL SECTIONS directive
  !$omp end parallel sections nowait

! 2.7.3 single-clause -> private-clause |
!                        firstprivate-clause
!   end-single-clause -> copyprivate-clause |
!                        nowait-clause

  !$omp parallel
  b = 1
  !ERROR: LASTPRIVATE clause is not allowed on the SINGLE directive
  !$omp single private(a) lastprivate(c)
  a = 3.14
  !ERROR: Clause NOWAIT is not allowed if clause COPYPRIVATE appears on the END SINGLE directive
  !ERROR: COPYPRIVATE variable 'a' may not appear on a PRIVATE or FIRSTPRIVATE clause on a SINGLE construct
  !ERROR: At most one NOWAIT clause can appear on the END SINGLE directive
  !$omp end single copyprivate(a) nowait nowait
  c = 2
  !$omp end parallel

! 2.7.4 workshare

  !$omp parallel
  !$omp workshare
  a = 1.0
  !$omp end workshare nowait
  !ERROR: NUM_THREADS clause is not allowed on the WORKSHARE directive
  !$omp workshare num_threads(4)
  a = 1.0
  !ERROR: COPYPRIVATE clause is not allowed on the END WORKSHARE directive
  !$omp end workshare nowait copyprivate(a)
  !$omp end parallel

! 2.8.1 simd-clause -> safelen-clause |
!                      simdlen-clause |
!                      linear-clause |
!                      aligned-clause |
!                      private-clause |
!                      lastprivate-clause |
!                      reduction-clause |
!                      collapse-clause

  a = 0.0
  !ERROR: TASK_REDUCTION clause is not allowed on the SIMD directive
  !$omp simd private(b) reduction(+:a) task_reduction(+:a)
  do i = 1, N
     a = a + b + 3.14
  enddo

  !ERROR: At most one SAFELEN clause can appear on the SIMD directive
  !$omp simd safelen(1) safelen(2)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the SIMDLEN clause must be a constant positive integer expression
  !$omp simd simdlen(-1)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the ALIGNED clause must be a constant positive integer expression
  !$omp simd aligned(b:-2)
  do i = 1, N
     a = 3.14
  enddo

  !$omp parallel
  !ERROR: The parameter of the SIMDLEN clause must be less than or equal to the parameter of the SAFELEN clause
  !$omp simd safelen(1+1) simdlen(1+2)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

! 2.11.1 parallel-do-clause -> parallel-clause |
!                              do-clause

  !ERROR: At most one PROC_BIND clause can appear on the PARALLEL DO directive
  !ERROR: A modifier may not be specified in a LINEAR clause on the PARALLEL DO directive
  !$omp parallel do proc_bind(master) proc_bind(close) linear(val(b))
  do i = 1, N
     a = 3.14
  enddo

! 2.8.3 do-simd-clause -> do-clause |
!                         simd-clause

  !$omp parallel
  !ERROR: No ORDERED clause with a parameter can be specified on the DO SIMD directive
  !ERROR: NOGROUP clause is not allowed on the DO SIMD directive
  !$omp do simd ordered(2) NOGROUP
  do i = 1, N
     do j = 1, N
        a = 3.14
     enddo
  enddo
  !$omp end parallel

! 2.11.4 parallel-do-simd-clause -> parallel-clause |
!                                   do-simd-clause

  !$omp parallel do simd collapse(2) safelen(2) &
  !$omp & simdlen(1) private(c) firstprivate(a) proc_bind(spread)
  do i = 1, N
     do j = 1, N
        a = 3.14
     enddo
  enddo

! 2.9.2 taskloop -> TASKLOOP [taskloop-clause[ [,] taskloop-clause]...]
!       taskloop-clause -> if-clause |
!                          shared-clause |
!                          private-clause |
!                          firstprivate-clause |
!                          lastprivate-clause |
!                          default-clause |
!                          grainsize-clause |
!                          num-tasks-clause |
!                          collapse-clause |
!                          final-clause |
!                          priority-clause |
!                          untied-clause |
!                          mergeable-clause |
!                          nogroup-clause

  !$omp taskloop
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: SCHEDULE clause is not allowed on the TASKLOOP directive
  !$omp taskloop schedule(static)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: GRAINSIZE and NUM_TASKS clauses are mutually exclusive and may not appear on the same TASKLOOP directive
  !$omp taskloop num_tasks(3) grainsize(2)
  do i = 1,N
     a = 3.14
  enddo

  !ERROR: At most one NUM_TASKS clause can appear on the TASKLOOP directive
  !ERROR: TASK_REDUCTION clause is not allowed on the TASKLOOP directive
  !$omp taskloop num_tasks(3) num_tasks(2) task_reduction(*:a)
  do i = 1,N
    a = 3.14
  enddo

! 2.13.1 master

  !$omp parallel
  !$omp master
  a=3.14
  !$omp end master
  !$omp end parallel

  !$omp parallel
  !ERROR: NUM_THREADS clause is not allowed on the MASTER directive
  !$omp master num_threads(4)
  a=3.14
  !$omp end master
  !$omp end parallel

! Standalone Directives (basic)

  !$omp taskyield
  !$omp barrier
  !$omp taskwait
  !$omp taskwait depend(source)
  ! !$omp taskwait depend(sink:i-1)
  ! !$omp target enter data map(to:arrayA) map(alloc:arrayB)
  ! !$omp target update from(arrayA) to(arrayB)
  ! !$omp target exit data map(from:arrayA) map(delete:arrayB)
  !$omp ordered depend(source)
  ! !$omp ordered depend(sink:i-1)
  !$omp flush (c)
  !$omp flush acq_rel
  !$omp flush release
  !$omp flush acquire
  !ERROR: If memory-order-clause is RELEASE, ACQUIRE, or ACQ_REL, list items must not be specified on the FLUSH directive
  !$omp flush release (c)
  !ERROR: SEQ_CST clause is not allowed on the FLUSH directive
  !$omp flush seq_cst
  !ERROR: RELAXED clause is not allowed on the FLUSH directive
  !$omp flush relaxed

  !$omp cancel DO
  !$omp cancellation point parallel

! 2.13.2 critical Construct

  ! !$omp critical (first)
  a = 3.14
  ! !$omp end critical (first)

! 2.9.1 task-clause -> if-clause |
!                      final-clause |
!                      untied-clause |
!                      default-clause |
!                      mergeable-clause |
!                      private-clause |
!                      firstprivate-clause |
!                      shared-clause |
!                      depend-clause |
!                      priority-clause

  !$omp task shared(a) default(none) if(task:a > 1.)
  a = 1.
  !$omp end task

  !ERROR: Unmatched directive name modifier TASKLOOP on the IF clause
  !$omp task private(a) if(taskloop:a.eq.1)
  a = 1.
  !$omp end task

  !ERROR: LASTPRIVATE clause is not allowed on the TASK directive
  !ERROR: At most one FINAL clause can appear on the TASK directive
  !$omp task lastprivate(b) final(a.GE.1) final(.false.)
  b = 1
  !$omp end task

  !ERROR: The parameter of the PRIORITY clause must be a positive integer expression
  !$omp task priority(-1) firstprivate(a) mergeable
  a = 3.14
  !$omp end task

! 2.9.3 taskloop-simd-clause -> taskloop-clause |
!                               simd-clause

  !$omp taskloop simd
  do i = 1, N
     a = 3.14
  enddo
  !$omp end taskloop simd

  !$omp taskloop simd reduction(+:a)
  do i = 1, N
     a = a + 3.14
  enddo
  !ERROR: Unmatched END TASKLOOP directive
  !$omp end taskloop

  !ERROR: GRAINSIZE and NUM_TASKS clauses are mutually exclusive and may not appear on the same TASKLOOP SIMD directive
  !$omp taskloop simd num_tasks(3) grainsize(2)
  do i = 1,N
     a = 3.14
  enddo

  !ERROR: The parameter of the SIMDLEN clause must be a constant positive integer expression
  !ERROR: The parameter of the ALIGNED clause must be a constant positive integer expression
  !$omp taskloop simd simdlen(-1) aligned(a:-2)
  do i = 1, N
     a = 3.14
  enddo
end program
