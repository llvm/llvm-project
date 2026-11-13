!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fopenmp -emit-llvm %s -o %t
! RUN: FileCheck %s --check-prefix=VIRTUAL < %t
! RUN: FileCheck %s --check-prefix=NOWAIT < %t
! RUN: %flang_fc1 -fopenmp -emit-llvm -mmlir -openmp-disable-virtual-collapse %s -o - | FileCheck %s --check-prefix=OFF --implicit-check-not=omp_collapsed.init
! RUN: %flang_fc1 -fopenmp -emit-llvm -mmlir -openmp-disable-virtual-collapse=true %s -o - | FileCheck %s --check-prefix=OFF --implicit-check-not=omp_collapsed.init
! RUN: %flang_fc1 -fopenmp -debug-info-kind=standalone -emit-llvm %s -o %t.debug.ll
! RUN: FileCheck %s --check-prefix=VIRTUAL < %t.debug.ll
! RUN: FileCheck %s --check-prefix=DBG < %t.debug.ll
! RUN: %flang_fc1 -fopenmp -emit-llvm -mmlir -openmp-disable-virtual-collapse=false %s -o - | FileCheck %s --check-prefix=VIRTUAL

! Range setup, one-time decoding and row exit retain the directive location,
! including its subprogram scope after parallel outlining.
! DBG-LABEL: define{{.*}} @static_worksharing(
! DBG-SAME: !dbg ![[WS_SCOPE:[0-9]+]] {
! DBG: call void @__kmpc_for_static_init_4u({{.*}}), !dbg ![[WS_LOC:[0-9]+]]
! DBG: omp_collapsed.init:
! DBG-NEXT: %omp_collapsed.range.begin = add i32 {{.*}}, 0, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.range.count = add i32 {{.*}}, 0, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG-NEXT: %[[NONEMPTY_32:[^ ]+]] = icmp ne i32 %omp_collapsed.range.count, 0, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG-NEXT: br i1 %[[NONEMPTY_32]], label %omp_collapsed.decode, label %omp_loop.exit, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG: omp_collapsed.decode:
! DBG-NEXT: %omp_collapsed.begin.1 = urem i32 %omp_collapsed.range.begin, 5, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.begin.0 = udiv i32 %omp_collapsed.range.begin, 5, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.row.remaining = add i32 %omp_collapsed.range.count, %omp_collapsed.begin.1, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG: omp_collapsed.row.done:
! DBG-NEXT: br i1 %omp_collapsed.row.is_final, label %omp_loop.exit, label %omp_loop.inc, !dbg ![[WS_LOC]]{{(,|$)}}
! DBG: {{^}}}
! DBG-LABEL: define{{.*}} @parallel_static..omp_par(
! DBG-SAME: !dbg ![[PAR_SCOPE:[0-9]+]] {
! DBG: call void @__kmpc_for_static_init_8u({{.*}}), !dbg ![[PAR_LOC:[0-9]+]]
! DBG: omp_collapsed.init:
! DBG-NEXT: %omp_collapsed.range.begin = add i64 {{.*}}, 0, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.range.count = add i64 {{.*}}, 0, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG-NEXT: %[[NONEMPTY_64:[^ ]+]] = icmp ne i64 %omp_collapsed.range.count, 0, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG-NEXT: br i1 %[[NONEMPTY_64]], label %omp_collapsed.decode, label %omp_loop.exit, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG: omp_collapsed.decode:
! DBG-NEXT: %omp_collapsed.begin.1 = urem i64 %omp_collapsed.range.begin, 3, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.begin.0 = udiv i64 %omp_collapsed.range.begin, 3, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG-NEXT: %omp_collapsed.row.remaining = add i64 %omp_collapsed.range.count, %omp_collapsed.begin.1, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG: omp_collapsed.row.done:
! DBG-NEXT: br i1 %omp_collapsed.row.is_final, label %omp_loop.exit, label %omp_loop.inc, !dbg ![[PAR_LOC]]{{(,|$)}}
! DBG: {{^}}}

! VIRTUAL-LABEL: define{{.*}} @static_worksharing(
! VIRTUAL: call void @__kmpc_for_static_init_4u
! VIRTUAL: omp_collapsed.init:
! VIRTUAL: omp_collapsed.decode:
! VIRTUAL: omp_collapsed.begin.1 = urem
! VIRTUAL: omp_collapsed.begin.0 = udiv
! VIRTUAL: omp_collapsed.row.remaining.0 = phi
! VIRTUAL-DAG: br i1 %omp_collapsed.row.is_final, label %omp_loop.exit, label %omp_loop.inc{{(,|$)}}
! VIRTUAL-DAG: call void @__kmpc_for_static_fini
! VIRTUAL-DAG: call void @__kmpc_barrier
! VIRTUAL: ret void
! OFF-LABEL: define{{.*}} @static_worksharing(
! OFF: call void @__kmpc_for_static_init_4u
! OFF-NOT: omp_collapsed.init
! OFF: urem
! OFF: udiv
! OFF: ret void
subroutine static_worksharing(a) bind(c)
  integer :: a(5,3), i, j, t
! DBG-DAG: ![[WS_LOC]] = !DILocation(line: [[#@LINE+1]], column: 9, scope: ![[WS_SCOPE]])
  !$omp do collapse(2) schedule(static) private(t)
  do j = 1, 3
    do i = 1, 5
      t = i + 5*j
      a(i,j) = t
    end do
  end do
  !$omp end do
end subroutine

! VIRTUAL-LABEL: define{{.*}} @parallel_static(
! VIRTUAL: call void {{.*}}@__kmpc_fork_call
! VIRTUAL: ret void
! VIRTUAL-LABEL: define internal void @parallel_static..omp_par(
! VIRTUAL: call void @__kmpc_for_static_init_8u
! VIRTUAL-DAG: omp_collapsed.init:
! VIRTUAL-DAG: call void @__kmpc_for_static_fini
! VIRTUAL-DAG: omp_collapsed.decode:
! VIRTUAL-DAG: omp_collapsed.row.done:{{.*[[:space:]]+}}br i1 %omp_collapsed.row.is_final, label %omp_loop.exit, label %omp_loop.inc{{(,|$)}}
! VIRTUAL: {{^}}}
! NOWAIT-LABEL: define internal void @parallel_static..omp_par(
! NOWAIT-NOT: @__kmpc_barrier
! NOWAIT: {{^}}}
! OFF-LABEL: define{{.*}} @parallel_static(
! OFF: call void {{.*}}@__kmpc_fork_call
! OFF: ret void
! OFF-LABEL: define internal void @parallel_static..omp_par(
! OFF: urem i64
! OFF: udiv i64
! OFF: ret void
! OFF: {{^}}}
subroutine parallel_static(a) bind(c)
  integer(kind=8) :: a(3,3), i, j
  !$omp parallel
! DBG-DAG: ![[PAR_LOC]] = !DILocation(line: [[#@LINE+1]], column: 9, scope: ![[PAR_SCOPE]])
  !$omp do collapse(2) schedule(static)
  do j = 9_8, 1_8, -4_8
    do i = 2_8, 6_8, 2_8
      a((i-2_8)/2_8+1_8,(9_8-j)/4_8+1_8) = 100_8*i+j
    end do
  end do
  !$omp end do nowait
  !$omp end parallel
end subroutine

! Check firstprivate input, reduction accumulation/combine and IV copyback
! through the Virtual executor, including its final-row exit.
! OFF-LABEL: define{{.*}} @static_clauses(
! VIRTUAL-LABEL: define{{.*}} @static_clauses(
! VIRTUAL-SAME: ptr %{{[^,]+}}, ptr [[SUM:%[^, ]+]], ptr [[SEED:%[^, ]+]], ptr [[II:%[^, ]+]], ptr [[JJ:%[^, )]+]])
! VIRTUAL: [[SEEDVALUE:%[^ ]+]] = load i32, ptr [[SEED]],
! VIRTUAL: store i32 [[SEEDVALUE]], ptr [[SEEDPRIVATE:%[^, ]+]],
! VIRTUAL: store i32 0, ptr [[REDUCTION:%[^, ]+]],
! VIRTUAL: store i32 5, ptr %p.upperbound,
! VIRTUAL: omp_collapsed.init:
! VIRTUAL: omp_collapsed.decode:
! VIRTUAL: [[OUTER_IV:%[^ ]+]] = phi i32 [ %omp_collapsed.begin.0, %omp_collapsed.decode ],
! VIRTUAL: [[ISTEP:%[^ ]+]] = mul nsw i32 [[OUTER_IV]], 1{{(,|$)}}
! VIRTUAL: [[ISOURCE:%[^ ]+]] = add nsw i32 [[ISTEP]], 1{{(,|$)}}
! VIRTUAL: [[INNER_IV:%[^ ]+]] = phi i32 [ %omp_collapsed.row.start.1.0,
! VIRTUAL: [[JSTEP:%[^ ]+]] = mul nsw i32 [[INNER_IV]], 1{{(,|$)}}
! VIRTUAL: [[JSOURCE:%[^ ]+]] = add nsw i32 [[JSTEP]], 1{{(,|$)}}
! VIRTUAL: [[COPYBACK:omp[.]loop_nest[.]region[0-9]+]]:{{ *}}; preds = %omp.loop_nest.region{{$}}
! VIRTUAL: store i32 [[INEXT:%[^, ]+]], ptr [[IPRIVATE:%[^, ]+]],
! VIRTUAL: store i32 [[JNEXT:%[^, ]+]], ptr [[JPRIVATE:%[^, ]+]],
! VIRTUAL: [[I:%[^ ]+]] = load i32, ptr [[IPRIVATE]],
! VIRTUAL: store i32 [[I]], ptr [[ILAST:%[^, ]+]],
! VIRTUAL: [[J:%[^ ]+]] = load i32, ptr [[JPRIVATE]],
! VIRTUAL: store i32 [[J]], ptr [[JLAST:%[^, ]+]],
! VIRTUAL: store i32 [[ISOURCE]], ptr [[IPRIVATE]],
! VIRTUAL: store i32 [[JSOURCE]], ptr [[JPRIVATE]],
! VIRTUAL: [[SEEDCOPY:%[^ ]+]] = load i32, ptr [[SEEDPRIVATE]],
! VIRTUAL: [[IBODY:%[^ ]+]] = load i32, ptr [[IPRIVATE]],
! VIRTUAL: [[ITEN:%[^ ]+]] = mul i32 [[IBODY]], 10{{(,|$)}}
! VIRTUAL: [[BASE:%[^ ]+]] = add i32 [[SEEDCOPY]], [[ITEN]]{{(,|$)}}
! VIRTUAL: [[JBODY:%[^ ]+]] = load i32, ptr [[JPRIVATE]],
! VIRTUAL: [[VALUE:%[^ ]+]] = add i32 [[BASE]], [[JBODY]]{{(,|$)}}
! VIRTUAL: store i32 [[VALUE]], ptr [[TPRIVATE:%[^, ]+]],
! VIRTUAL: [[ACC:%[^ ]+]] = load i32, ptr [[REDUCTION]],
! VIRTUAL: [[TERM:%[^ ]+]] = load i32, ptr [[TPRIVATE]],
! VIRTUAL: [[NEXT:%[^ ]+]] = add i32 [[ACC]], [[TERM]]{{(,|$)}}
! VIRTUAL: store i32 [[NEXT]], ptr [[REDUCTION]],
! VIRTUAL: [[INEXT]] = add i32 [[ISOURCE]], 1{{(,|$)}}
! VIRTUAL: [[ILASTITER:%[^ ]+]] = icmp sgt i32 [[INEXT]], 2{{(,|$)}}
! VIRTUAL: [[JNEXT]] = add i32 [[JSOURCE]], 1{{(,|$)}}
! VIRTUAL: [[JLASTITER:%[^ ]+]] = icmp sgt i32 [[JNEXT]], 3{{(,|$)}}
! VIRTUAL: [[LAST:%[^ ]+]] = and i1 [[ILASTITER]], [[JLASTITER]]{{(,|$)}}
! VIRTUAL: br i1 [[LAST]], label %[[COPYBACK]],
! VIRTUAL: omp_collapsed.row.done:
! VIRTUAL-NEXT: br i1 %omp_collapsed.row.is_final, label %[[EXIT:omp_loop.exit]], label %omp_loop.inc{{(,|$)}}
! VIRTUAL: [[EXIT]]:
! VIRTUAL-NEXT: call void @__kmpc_for_static_fini
! VIRTUAL: call i32 @__kmpc_reduce(
! VIRTUAL: [[IFINAL:%[^ ]+]] = load i32, ptr [[ILAST]],
! VIRTUAL: store i32 [[IFINAL]], ptr [[II]],
! VIRTUAL: [[JFINAL:%[^ ]+]] = load i32, ptr [[JLAST]],
! VIRTUAL: store i32 [[JFINAL]], ptr [[JJ]],
! VIRTUAL: ret void
! VIRTUAL: [[ORIGINAL:%[^ ]+]] = load i32, ptr [[SUM]],
! VIRTUAL: [[PARTIAL:%[^ ]+]] = load i32, ptr [[REDUCTION]],
! VIRTUAL: [[RESULT:%[^ ]+]] = add i32 [[ORIGINAL]], [[PARTIAL]]{{(,|$)}}
! VIRTUAL: store i32 [[RESULT]], ptr [[SUM]],
! VIRTUAL: call void @__kmpc_end_reduce(
subroutine static_clauses(a, s, seed, ii, jj) bind(c)
  integer :: a(6), s, seed, ii, jj, i, j, t
  !$omp do collapse(2) schedule(static) reduction(+:s) firstprivate(seed) lastprivate(i,j) private(t)
  do i = 1, 2
    do j = 1, 3
      t = seed+10*i+j
      a((i-1)*3+j) = t
      s = s+t
    end do
  end do
  !$omp end do
  ii = i
  jj = j
end subroutine

! VIRTUAL-LABEL: define{{.*}} @static_copyback64(
! VIRTUAL-SAME: ptr %{{[^,]+}}, ptr [[II:%[^, ]+]], ptr [[JJ:%[^, )]+]])
! VIRTUAL: store i64 8, ptr %p.upperbound,
! VIRTUAL: omp_collapsed.init:
! VIRTUAL: omp_collapsed.decode:
! VIRTUAL: [[OUTER_IV:%[^ ]+]] = phi i64 [ %omp_collapsed.begin.0, %omp_collapsed.decode ],
! VIRTUAL: [[ISTEP:%[^ ]+]] = mul nsw i64 [[OUTER_IV]], 2{{(,|$)}}
! VIRTUAL: [[ISOURCE:%[^ ]+]] = add nsw i64 [[ISTEP]], 2{{(,|$)}}
! VIRTUAL: [[INNER_IV:%[^ ]+]] = phi i64 [ %omp_collapsed.row.start.1.0,
! VIRTUAL: [[JSTEP:%[^ ]+]] = mul nsw i64 [[INNER_IV]], -4{{(,|$)}}
! VIRTUAL: [[JSOURCE:%[^ ]+]] = add nsw i64 [[JSTEP]], 9{{(,|$)}}
! VIRTUAL: [[COPYBACK:omp[.]loop_nest[.]region[0-9]+]]:{{ *}}; preds = %omp.loop_nest.region{{$}}
! VIRTUAL: store i64 [[INEXT:%[^, ]+]], ptr [[IPRIVATE:%[^, ]+]],
! VIRTUAL: store i64 [[JNEXT:%[^, ]+]], ptr [[JPRIVATE:%[^, ]+]],
! VIRTUAL: [[I:%[^ ]+]] = load i64, ptr [[IPRIVATE]],
! VIRTUAL: store i64 [[I]], ptr [[ILAST:%[^, ]+]],
! VIRTUAL: [[J:%[^ ]+]] = load i64, ptr [[JPRIVATE]],
! VIRTUAL: store i64 [[J]], ptr [[JLAST:%[^, ]+]],
! VIRTUAL: store i64 [[ISOURCE]], ptr [[IPRIVATE]],
! VIRTUAL: store i64 [[JSOURCE]], ptr [[JPRIVATE]],
! VIRTUAL: [[IBODY:%[^ ]+]] = load i64, ptr [[IPRIVATE]],
! VIRTUAL: [[ITERM:%[^ ]+]] = mul i64 [[IBODY]], 100{{(,|$)}}
! VIRTUAL: [[JBODY:%[^ ]+]] = load i64, ptr [[JPRIVATE]],
! VIRTUAL: [[VALUE:%[^ ]+]] = add i64 [[ITERM]], [[JBODY]]{{(,|$)}}
! VIRTUAL: store i64 [[VALUE]], ptr %{{[^, ]+}},
! VIRTUAL: [[INEXT]] = add i64 [[ISOURCE]], 2{{(,|$)}}
! VIRTUAL: [[ILASTITER:%[^ ]+]] = icmp sgt i64 [[INEXT]], 6{{(,|$)}}
! VIRTUAL: [[JNEXT]] = add i64 [[JSOURCE]], -4{{(,|$)}}
! VIRTUAL: [[JLASTITER:%[^ ]+]] = icmp slt i64 [[JNEXT]], 1{{(,|$)}}
! VIRTUAL: [[LAST:%[^ ]+]] = and i1 [[ILASTITER]], [[JLASTITER]]{{(,|$)}}
! VIRTUAL: br i1 [[LAST]], label %[[COPYBACK]],
! VIRTUAL: omp_collapsed.row.done:
! VIRTUAL-NEXT: br i1 %omp_collapsed.row.is_final, label %[[EXIT:omp_loop.exit]], label %omp_loop.inc{{(,|$)}}
! VIRTUAL: [[EXIT]]:
! VIRTUAL-NEXT: call void @__kmpc_for_static_fini
! VIRTUAL: [[IFINAL:%[^ ]+]] = load i64, ptr [[ILAST]],
! VIRTUAL: store i64 [[IFINAL]], ptr [[II]],
! VIRTUAL: [[JFINAL:%[^ ]+]] = load i64, ptr [[JLAST]],
! VIRTUAL: store i64 [[JFINAL]], ptr [[JJ]],
! VIRTUAL: ret void
subroutine static_copyback64(a, ii, jj) bind(c)
  integer(kind=8) :: a(9), ii, jj, i, j
  !$omp do collapse(2) schedule(static) lastprivate(i,j)
  do i = 2_8, 6_8, 2_8
    do j = 9_8, 1_8, -4_8
      a(((i-2_8)/2_8)*3_8+(9_8-j)/4_8+1_8) = 100_8*i+j
    end do
  end do
  !$omp end do
  ii = i
  jj = j
end subroutine
