!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %if x86-registered-target %{ %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s %}
!RUN: %if aarch64-registered-target %{ %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s %}

! Int "==" → cmpxchg, default (monotonic) ordering
!CHECK-LABEL: define void @atomic_compare_integer_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
subroutine atomic_compare_integer(x, e, d)
  integer :: x, e, d
  !$omp atomic compare
  if (x == e) x = d
end

! seq_cst ordering → cmpxchg seq_cst + flush
!CHECK-LABEL: define void @atomic_compare_seq_cst_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] seq_cst seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_seq_cst(x, e, d)
  integer :: x, e, d
  !$omp atomic compare seq_cst
  if (x == e) x = d
end

! acquire ordering on compare (update) is valid in OpenMP 5.1
!CHECK-LABEL: define void @atomic_compare_acquire_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] acquire acquire
subroutine atomic_compare_acquire(x, e, d)
  integer :: x, e, d
  !$omp atomic compare acquire
  if (x == e) x = d
end

! release ordering → cmpxchg release + flush
!CHECK-LABEL: define void @atomic_compare_release_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] release monotonic
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_release(x, e, d)
  integer :: x, e, d
  !$omp atomic compare release
  if (x == e) x = d
end

! relaxed ordering → cmpxchg monotonic
!CHECK-LABEL: define void @atomic_compare_relaxed_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]], align 4
!CHECK: cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
subroutine atomic_compare_relaxed(x, e, d)
  integer :: x, e, d
  !$omp atomic compare relaxed
  if (x == e) x = d
end

! Less-than comparison → atomicrmw max (signed)
!CHECK-LABEL: define void @atomic_compare_lt_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw max ptr %[[X]], i32 %[[EVAL]] monotonic
subroutine atomic_compare_lt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x < e) x = e
end

! Less-than with seq_cst → atomicrmw max seq_cst + flush (signed)
!CHECK-LABEL: define void @atomic_compare_lt_seq_cst_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw max ptr %[[X]], i32 %[[EVAL]] seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_lt_seq_cst(x, e)
  integer :: x, e
  !$omp atomic compare seq_cst
  if (x < e) x = e
end

! Less-than with acquire on compare (update) is valid in OpenMP 5.1
!CHECK-LABEL: define void @atomic_compare_lt_acquire_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw max ptr %[[X]], i32 %[[EVAL]] acquire
subroutine atomic_compare_lt_acquire(x, e)
  integer :: x, e
  !$omp atomic compare acquire
  if (x < e) x = e
end

! Greater-than comparison → atomicrmw min (signed)
!CHECK-LABEL: define void @atomic_compare_gt_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]], align 4
!CHECK: atomicrmw min ptr %[[X]], i32 %[[EVAL]] monotonic
subroutine atomic_compare_gt(x, e)
  integer :: x, e
  !$omp atomic compare
  if (x > e) x = e
end

! Real "==" → NaN guard + ±0.0 guard + cmpxchg
! IEEE 754 special cases for cmpxchg (which is bitwise):
!   1. NaN != NaN but identical NaN bit patterns would match → skip cmpxchg
!   2. -0.0 == +0.0 but different bit patterns → use loaded bit-pattern
!CHECK-LABEL: define void @atomic_compare_real_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EVAL:.*]] = load float, ptr %[[E]], align 4
!CHECK: %[[DVAL:.*]] = load float, ptr %[[D]], align 4
!CHECK: %[[EBC:.*]] = bitcast float %[[EVAL]] to i32
!CHECK: %[[DBC:.*]] = bitcast float %[[DVAL]] to i32
!CHECK: load atomic i32, ptr %[[X]] monotonic
!CHECK: %[[EISNAN:.*]] = fcmp uno float %[[EVAL]], %[[EVAL]]
!CHECK: %[[XISNAN:.*]] = fcmp uno float %{{.*}}, %{{.*}}
!CHECK: %[[EITHERNAN:.*]] = or i1 %[[EISNAN]], %[[XISNAN]]
!CHECK: br i1 %[[EITHERNAN]], label %[[NANBB:[^,]+]], label %[[NOTNANBB:[^,]+]]
!CHECK: [[NANBB]]:
!CHECK-NEXT: br label %[[EXIT:[^ ]+]]
!CHECK: [[NOTNANBB]]:
!CHECK: %[[XISZERO:.*]] = fcmp oeq float %{{.*}}, 0.000000e+00
!CHECK: %[[EISZERO:.*]] = fcmp oeq float %[[EVAL]], 0.000000e+00
!CHECK: %[[BOTH:.*]] = and i1 %[[XISZERO]], %[[EISZERO]]
!CHECK: br i1 %[[BOTH]], label %[[ZERO:[^,]+]], label %[[NORMAL:[^,]+]]
!CHECK: [[ZERO]]:
!CHECK: cmpxchg ptr %[[X]], i32 %{{.*}}, i32 %[[DBC]] monotonic monotonic
!CHECK: br label %[[EXIT]]
!CHECK: [[NORMAL]]:
!CHECK: cmpxchg ptr %[[X]], i32 %[[EBC]], i32 %[[DBC]] monotonic monotonic
!CHECK: br label %[[EXIT]]
subroutine atomic_compare_real(x, e, d)
  real :: x, e, d
  !$omp atomic compare
  if (x == e) x = d
end

! Complex(4) equality → type-punned i64 cmpxchg with consistent alignment
!CHECK-LABEL: define void @atomic_compare_complex4_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EALLOCA:.*]] = alloca { float, float }, align [[ALIGN:[0-9]+]]
!CHECK: %[[DALLOCA:.*]] = alloca { float, float }, align [[ALIGN]]
!CHECK: store { float, float } %{{.*}}, ptr %[[EALLOCA]], align [[ALIGN]]
!CHECK: %[[EINT:.*]] = load i64, ptr %[[EALLOCA]], align [[ALIGN]]
!CHECK: store { float, float } %{{.*}}, ptr %[[DALLOCA]], align [[ALIGN]]
!CHECK: %[[DINT:.*]] = load i64, ptr %[[DALLOCA]], align [[ALIGN]]
!CHECK: cmpxchg ptr %[[X]], i64 %[[EINT]], i64 %[[DINT]] monotonic monotonic, align [[ALIGN]]
subroutine atomic_compare_complex4(x, e, d)
  complex :: x, e, d
  !$omp atomic compare
  if (x == e) x = d
end

! Complex(8) equality → type-punned i128 cmpxchg with consistent alignment
!CHECK-LABEL: define void @atomic_compare_complex8_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: %[[EALLOCA:.*]] = alloca { double, double }, align [[ALIGN:[0-9]+]]
!CHECK: %[[DALLOCA:.*]] = alloca { double, double }, align [[ALIGN]]
!CHECK: store { double, double } %{{.*}}, ptr %[[EALLOCA]], align [[ALIGN]]
!CHECK: %[[EINT:.*]] = load i128, ptr %[[EALLOCA]], align [[ALIGN]]
!CHECK: store { double, double } %{{.*}}, ptr %[[DALLOCA]], align [[ALIGN]]
!CHECK: %[[DINT:.*]] = load i128, ptr %[[DALLOCA]], align [[ALIGN]]
!CHECK: cmpxchg ptr %[[X]], i128 %[[EINT]], i128 %[[DINT]] monotonic monotonic, align [[ALIGN]]
subroutine atomic_compare_complex8(x, e, d)
  complex(8) :: x, e, d
  !$omp atomic compare
  if (x == e) x = d
end

! Complex(4) equality with seq_cst → type-punned i64 cmpxchg seq_cst + flush
!CHECK-LABEL: define void @atomic_compare_complex4_seq_cst_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: cmpxchg ptr %[[X]], i64 %{{.*}}, i64 %{{.*}} seq_cst seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_complex4_seq_cst(x, e, d)
  complex :: x, e, d
  !$omp atomic compare seq_cst
  if (x == e) x = d
end

! CHECK-LABEL: define void @omp_atomic_compare_ptr_
! CHECK-SAME: (ptr %[[X_DESC_ARG:.*]], ptr %[[E_DESC_ARG:.*]], ptr %[[D_DESC_ARG:.*]])

! CHECK: %[[D_COPY:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[E_COPY:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[X_COPY:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[D_TMP_DESC:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[E_TMP_DESC:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[X_TMP_DESC:.*]] = alloca { ptr, {{.*}} }
! CHECK: %[[TX:.*]] = alloca i32
! CHECK: %[[TE:.*]] = alloca i32
! CHECK: %[[TD:.*]] = alloca i32

! CHECK: %[[X_INIT:.*]] = insertvalue { ptr, {{.*}} } {{.*}}, ptr %[[TX]], 0
! CHECK: store { ptr, {{.*}} } %[[X_INIT]], ptr %[[X_TMP_DESC]]
! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[X_DESC_ARG]], ptr{{.*}} %[[X_TMP_DESC]], {{.*}})

! CHECK: %[[E_INIT:.*]] = insertvalue { ptr, {{.*}} } {{.*}}, ptr %[[TE]], 0
! CHECK: store { ptr, {{.*}} } %[[E_INIT]], ptr %[[E_TMP_DESC]]
! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[E_DESC_ARG]], ptr{{.*}} %[[E_TMP_DESC]], {{.*}})

! CHECK: %[[D_INIT:.*]] = insertvalue { ptr, {{.*}} } {{.*}}, ptr %[[TD]], 0
! CHECK: store { ptr, {{.*}} } %[[D_INIT]], ptr %[[D_TMP_DESC]]
! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[D_DESC_ARG]], ptr{{.*}} %[[D_TMP_DESC]], {{.*}})

! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[X_COPY]], ptr{{.*}} %[[X_DESC_ARG]], {{.*}})
! CHECK: %[[X_FIELD:.*]] = getelementptr { ptr, {{.*}} }, ptr %[[X_COPY]], {{.*}}
! CHECK: %[[X_ADDR:.*]] = load ptr, ptr %[[X_FIELD]]

! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[E_COPY]], ptr{{.*}} %[[E_DESC_ARG]], {{.*}})
! CHECK: %[[E_FIELD:.*]] = getelementptr { ptr, {{.*}} }, ptr %[[E_COPY]], {{.*}}
! CHECK: %[[E_ADDR:.*]] = load ptr, ptr %[[E_FIELD]]
! CHECK: %[[E_VAL:.*]] = load i32, ptr %[[E_ADDR]]

! CHECK: call void @llvm.memcpy.{{.*}}(ptr{{.*}} %[[D_COPY]], ptr{{.*}} %[[D_DESC_ARG]], {{.*}})
! CHECK: %[[D_FIELD:.*]] = getelementptr { ptr, {{.*}} }, ptr %[[D_COPY]], {{.*}}
! CHECK: %[[D_ADDR:.*]] = load ptr, ptr %[[D_FIELD]]
! CHECK: %[[D_VAL:.*]] = load i32, ptr %[[D_ADDR]]

! CHECK: cmpxchg ptr %[[X_ADDR]], i32 %[[E_VAL]], i32 %[[D_VAL]] monotonic monotonic{{.*}}
! CHECK: ret void
subroutine omp_atomic_compare_ptr(x, e, d)
  implicit none
  integer, target :: tx, te, td
  integer, pointer :: x, e, d

  x => tx
  e => te
  d => td

  !$omp atomic compare
    if (x == e) x = d
  !$omp end atomic

end subroutine

! Complex(4) equality with seq_cst → type-punned i64 cmpxchg seq_cst + flush
!CHECK-LABEL: define void @atomic_compare_weak_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]])
!CHECK: cmpxchg weak ptr %[[X]], i32 %{{.*}}, i32 %{{.*}} seq_cst seq_cst
!CHECK: call void @__kmpc_flush(
subroutine atomic_compare_weak(x, e, d)
  integer :: x, e, d
  !$omp atomic compare weak seq_cst
  if (x == e) x = d
end 

! Integer equality compare+capture (prefix): v=x; if(x==e) x=d
! v captures old value of x
!CHECK-LABEL: define void @atomic_compare_capture_int_eq_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]]
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]]
!CHECK: %[[RES:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
!CHECK: %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
!CHECK: store i32 %[[OLD]], ptr %[[V]]
subroutine atomic_compare_capture_int_eq(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    v = x
    if (x == e) x = d
  !$omp end atomic
end

! Compare+capture with clause order reversed: capture compare (still prefix read)
!CHECK-LABEL: define void @atomic_capture_compare_int_eq_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]]
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]]
!CHECK: %[[RES:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
!CHECK: %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
!CHECK: store i32 %[[OLD]], ptr %[[V]]
subroutine atomic_capture_compare_int_eq(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic capture compare
    v = x
    if (x == e) x = d
  !$omp end atomic
end

! Postfix compare+capture: if (x == e) x = d; v = x
! v captures new value (d if swapped, old x if not)
!CHECK-LABEL: define void @atomic_compare_capture_postfix_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]]
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]]
!CHECK: %[[RES:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
!CHECK: %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
!CHECK: %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[RES]], 1
!CHECK: %[[NEWVAL:.*]] = select i1 %[[SUCCESS]], i32 %[[DVAL]], i32 %[[OLD]]
!CHECK: store i32 %[[NEWVAL]], ptr %[[V]]
subroutine atomic_compare_capture_postfix(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    if (x == e) x = d
    v = x
  !$omp end atomic
end

! Fail-only compare+capture: if (x == e) x = d; else v = x
! v is only written when the comparison fails
!CHECK-LABEL: define void @atomic_compare_capture_fail_only_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: %[[EVAL:.*]] = load i32, ptr %[[E]]
!CHECK: %[[DVAL:.*]] = load i32, ptr %[[D]]
!CHECK: %[[RES:.*]] = cmpxchg ptr %[[X]], i32 %[[EVAL]], i32 %[[DVAL]] monotonic monotonic
!CHECK: %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
!CHECK: %[[SUCCESS:.*]] = extractvalue { i32, i1 } %[[RES]], 1
!CHECK: br i1 %[[SUCCESS]], label %[[EXIT:.*]], label %[[CONT:.*]]
!CHECK: {{.*}}:
!CHECK: store i32 %[[OLD]], ptr %[[V]]
!CHECK: br label %[[EXIT2:.*]]
subroutine atomic_compare_capture_fail_only(x, e, d, v)
  integer :: x, e, d, v
  !$omp atomic compare capture
    if (x == e) then
      x = d
    else
      v = x
    end if
  !$omp end atomic
end

! Real equality compare+capture (postfix): if(x==e) x=d; v=x
! v captures new value of x (d if success, old if fail)
!CHECK-LABEL: define void @atomic_compare_capture_real_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: %[[EVAL:.*]] = load float, ptr %[[E]]{{.*}}
!CHECK: %[[DVAL:.*]] = load float, ptr %[[D]]{{.*}}
!CHECK: %[[EBITS:.*]] = bitcast float %[[EVAL]] to i32
!CHECK: %[[DBITS:.*]] = bitcast float %[[DVAL]] to i32
!CHECK: %[[XLOAD:.*]] = load atomic i32, ptr %[[X]] monotonic{{.*}}
!CHECK: %[[XFP:.*]] = bitcast i32 %[[XLOAD]] to float
! Part 1: NaN check - if either x or e is NaN, comparison fails
!CHECK: %[[E_ISNAN:.*]] = fcmp uno float %[[EVAL]], %[[EVAL]]
!CHECK: %[[X_ISNAN:.*]] = fcmp uno float %[[XFP]], %[[XFP]]
!CHECK: %[[EITHER_NAN:.*]] = or i1 %[[E_ISNAN]], %[[X_ISNAN]]
!CHECK: br i1 %[[EITHER_NAN]], label %[[NAN_BB:.*]], label %[[NOTNAN_BB:.*]]
!CHECK: [[NAN_BB]]:
!CHECK: br label %[[EXIT_BB:.*]]
! Part 2: Both-zero check - handles +0.0 vs -0.0 (same value, different bits)
!CHECK: [[NOTNAN_BB]]:
!CHECK: %[[XISZERO:.*]] = fcmp oeq float %[[XFP]], 0.000000e+00
!CHECK: %[[EISZERO:.*]] = fcmp oeq float %[[EVAL]], 0.000000e+00
!CHECK: %[[BOTHZERO:.*]] = and i1 %[[XISZERO]], %[[EISZERO]]
!CHECK: br i1 %[[BOTHZERO]], label %[[ZERO_BB:.*]], label %[[NORMAL_BB:.*]]
!CHECK: [[ZERO_BB]]:
!CHECK: %[[ZERORES:.*]] = cmpxchg ptr %[[X]], i32 %[[XLOAD]], i32 %[[DBITS]] monotonic monotonic{{.*}}
!CHECK: br label %[[EXIT_BB]]
! Part 3: Normal compare - standard cmpxchg with bitcasted expected value
!CHECK: [[NORMAL_BB]]:
!CHECK: %[[NORMRES:.*]] = cmpxchg ptr %[[X]], i32 %[[EBITS]], i32 %[[DBITS]] monotonic monotonic{{.*}}
!CHECK: br label %[[EXIT_BB]]
! Exit: select v = (success ? d : old_x)
!CHECK: [[EXIT_BB]]:
!CHECK: %[[OLD:.*]] = phi i32 {{.*}}
!CHECK: %[[OK:.*]] = phi i1 {{.*}}
!CHECK: %[[OLDFP:.*]] = bitcast i32 %[[OLD]] to float
!CHECK: %[[VVAL:.*]] = select i1 %[[OK]], float %[[DVAL]], float %[[OLDFP]]
!CHECK: store float %[[VVAL]], ptr %[[V]]{{.*}}
subroutine atomic_compare_capture_real(x, e, d, v)
  real :: x, e, d, v
  !$omp atomic compare capture
    if (x == e) x = d
    v = x
  !$omp end atomic
end

! Logical .eqv. compare+capture (postfix): if(x.eqv.e) x=d; v=x
! Logicals are compared as integers after truthiness normalization
!CHECK-LABEL: define void @atomic_compare_capture_logical_(
!CHECK-SAME: ptr noalias %[[X:.*]], ptr noalias %[[E:.*]], ptr noalias %[[D:.*]], ptr noalias %[[V:.*]])
!CHECK: cmpxchg ptr %[[X]], i32 %{{.*}}, i32 %{{.*}} monotonic monotonic{{.*}}
subroutine atomic_compare_capture_logical(x, e, d, v)
  logical :: x, e, d, v
  !$omp atomic compare capture
    if (x .eqv. e) x = d
    v = x
  !$omp end atomic
end

! Logical compare+capture inside a parallel region.
!CHECK-LABEL: define void @atomic_compare_capture_logical_parallel_(
!CHECK-SAME: ptr noalias %[[X:.*]])
!CHECK: %[[CGEP:.*]] = getelementptr { ptr }, ptr %structArg, i32 0, i32 0
!CHECK: store ptr %[[X]], ptr %[[CGEP]]
!CHECK: call void {{.*}}@__kmpc_fork_call{{.*}}@atomic_compare_capture_logical_parallel_..omp_par
!CHECK-LABEL: define internal void @atomic_compare_capture_logical_parallel_..omp_par(
!CHECK-SAME: ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %[[STRUCTARG:.*]])
!CHECK: %[[GEP:.*]] = getelementptr { ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
!CHECK: %[[SHARED:.*]] = load ptr, ptr %[[GEP]]
!CHECK: %[[RES:.*]] = cmpxchg ptr %[[SHARED]], i32 %{{.*}}, i32 %{{.*}} monotonic monotonic{{.*}}
!CHECK: %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
!CHECK: store i32 %[[OLD]], ptr %{{.*}}
subroutine atomic_compare_capture_logical_parallel(x)
  logical :: x, expected, desired, old_value
  !$omp parallel private(old_value, expected, desired)
    expected = .true.
    desired  = .false.
    !$omp atomic compare capture
      old_value = x
      if (x .eqv. expected) then
        x = desired
      end if
    !$omp end atomic
  !$omp end parallel
end

