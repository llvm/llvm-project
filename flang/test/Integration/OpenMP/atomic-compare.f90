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
