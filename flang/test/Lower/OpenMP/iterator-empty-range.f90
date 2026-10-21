! RUN: %flang_fc1 -emit-llvm -O1 -fopenmp -fopenmp-version=52 -o - %s \
! RUN:   | FileCheck %s

! Empty ranges must produce zero entries and skip locator evaluation. Test
! dynamic bounds and both step directions. Bounds equal to the initial value
! must still produce one entry. Constant empty ranges are tested below.
subroutine depend_empty_positive(a, m, d)
  integer :: a(8), m, d
  !$omp task depend(iterator(i=3:m), in: a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_positive_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[COUNT]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_positive(a, m, d)
  integer :: a(8), m, d
  !$omp task affinity(iterator(i=3:m): a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_positive_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[COUNT]]
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[COUNT]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine depend_empty_negative(a, m, d)
  integer :: a(8), m, d
  !$omp task depend(iterator(i=3:m:-2), in: a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_negative_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i33
! CHECK: [[SPAN:%.*]] = sub nsw i33 3, [[WIDE]]
! CHECK: [[EMPTY:%.*]] = icmp sgt i32 [[M]], 3
! CHECK: [[HALF:%.*]] = lshr i33 [[SPAN]], 1
! CHECK: [[NONEMPTY33:%.*]] = add {{.*}}i33 [[HALF]], 1
! CHECK: [[NONEMPTY:%.*]] = zext i33 [[NONEMPTY33]] to i64
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[COUNT]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK-NEXT: br i1 [[EMPTY]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_negative(a, m, d)
  integer :: a(8), m, d
  !$omp task affinity(iterator(i=3:m:-2): a(i/d))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_negative_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[WIDE:%.*]] = sext i32 [[M]] to i33
! CHECK: [[SPAN:%.*]] = sub nsw i33 3, [[WIDE]]
! CHECK: [[EMPTY:%.*]] = icmp sgt i32 [[M]], 3
! CHECK: [[HALF:%.*]] = lshr i33 [[SPAN]], 1
! CHECK: [[NONEMPTY33:%.*]] = add {{.*}}i33 [[HALF]], 1
! CHECK: [[NONEMPTY:%.*]] = zext i33 [[NONEMPTY33]] to i64
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 [[NONEMPTY]]
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[COUNT]]
! CHECK: br i1 [[EMPTY]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

! A zero count in either retained dimension must suppress the entire locator.
subroutine depend_empty_product(a, m, n, d)
  integer :: a(8,8), m, n, d
  !$omp task depend(iterator(i=3:m, j=1:n), in: a(i/d,j))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @depend_empty_product_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[N:%.*]] = load i32, ptr
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: [[NCOUNT:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[N]], i32 0)
! CHECK: [[N64:%.*]] = zext {{.*}}i32 [[NCOUNT]] to i64
! CHECK: [[TOTAL:%.*]] = mul {{.*}}i64 [[COUNT]], [[N64]]
! CHECK: [[SIZE:%.*]] = mul {{.*}}i64 [[TOTAL]], 24
! CHECK: [[LIST:%.*]] = {{.*}}call ptr @malloc(i64 [[SIZE]])
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[TOTAL]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[TOTAL]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

subroutine affinity_empty_product(a, m, n, d)
  integer :: a(8,8), m, n, d
  !$omp task affinity(iterator(i=3:m, j=1:n): a(i/d,j))
  !$omp end task
end
! CHECK-LABEL: define {{.*}} @affinity_empty_product_(
! CHECK: [[M:%.*]] = load i32, ptr
! CHECK: [[N:%.*]] = load i32, ptr
! CHECK: [[EMPTY:%.*]] = icmp slt i32 [[M]], 3
! CHECK: [[COUNT:%.*]] = select i1 [[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: [[NCOUNT:%.*]] = {{.*}}call i32 @llvm.smax.i32(i32 [[N]], i32 0)
! CHECK: [[N64:%.*]] = zext {{.*}}i32 [[NCOUNT]] to i64
! CHECK: [[TOTAL:%.*]] = mul {{.*}}i64 [[COUNT]], [[N64]]
! CHECK: [[LIST:%.*]] = alloca { i64, i64, i32 }, i64 [[TOTAL]]
! CHECK: [[ZERO:%.*]] = icmp eq i64 [[TOTAL]], 0
! CHECK-NEXT: br i1 [[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32
! CHECK: [[CONT]]:
! CHECK: [[RUNTIME_COUNT:%.*]] = trunc i64 [[TOTAL]] to i32
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}},
! CHECK-SAME: i32 [[RUNTIME_COUNT]], ptr {{[^,)]*}}[[LIST]]

! Unused iterator ranges must not affect locator counts.

! The unused j range must not affect the number of dependences, including
! when m < 3 makes that range empty.
subroutine depend_unused_iterator(a, m)
  integer :: a(2), m

  !$omp task depend(iterator(i = 1:2, j = 3:m), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_unused_iterator_(
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 2,

! Affinity uses the same per-locator selection and must also ignore empty j.
subroutine affinity_unused_iterator(a, m)
  integer :: a(2), m

  !$omp task affinity(iterator(i = 1:2, j = 3:m): a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @affinity_unused_iterator_(
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 2,

! Source references lost during folding must still control evaluation.

! The folded-away i occurrence still makes an empty i range suppress the
! locator, including its division by d. Test both consumers of omp.iterator.
subroutine depend_folded_empty(a, m, d)
  integer :: a(2), m, d
  !$omp task depend(iterator(i = 1:m, j = 1:2), in: a(j/d + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_folded_empty_(
! CHECK: %[[M:.*]] = load i32,
! CHECK: %[[N:.*]] = {{.*}}call i32 @llvm.smax.i32(i32 %[[M]], i32 0)
! CHECK: %[[COUNT:.*]] = shl{{.*}} i32 %[[N]], 1
! CHECK: %[[TOTAL:.*]] = zext i32 %[[COUNT]] to i64
! CHECK: %[[SIZE:.*]] = mul{{.*}} i64 %[[TOTAL]], {{[0-9]+}}
! CHECK: call ptr @malloc(i64 %[[SIZE]])
! CHECK-NOT: sdiv
! CHECK: %[[EMPTY:.*]] = icmp slt i32 %[[M]], 1
! CHECK: br i1 %[[EMPTY]], label %[[EXIT:.*]], label %[[PRE:.*]]
! CHECK: [[PRE]]:
! CHECK: %[[D:.*]] = load i32,
! CHECK: br label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32 %{{.*}}, %[[D]]
! CHECK: [[EXIT]]:
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 %[[COUNT]],

! A statically empty referenced range must also skip locator evaluation.
subroutine depend_folded_zero(a, d)
  integer :: a(2), d
  !$omp task depend(iterator(i = 1:0, j = 1:2), in: a(j/d + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_folded_zero_(
! CHECK-NOT: sdiv
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 0,
! CHECK-NOT: sdiv
! CHECK: ret void

subroutine affinity_folded_zero(a, d)
  integer :: a(2), d
  !$omp task affinity(iterator(i = 1:0, j = 1:2): a(j/d + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @affinity_folded_zero_(
! CHECK-NOT: sdiv
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 0,
! CHECK-NOT: sdiv
! CHECK: ret void

subroutine affinity_folded_empty(a, m, d)
  integer :: a(2), m, d
  !$omp task affinity(iterator(i = 1:m, j = 1:2): a(j/d + 0*i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @affinity_folded_empty_(
! CHECK: %[[M:.*]] = load i32,
! CHECK: %[[N:.*]] = {{.*}}call i32 @llvm.smax.i32(i32 %[[M]], i32 0)
! CHECK: %[[COUNT:.*]] = shl{{.*}} i32 %[[N]], 1
! CHECK: %[[TOTAL:.*]] = zext i32 %[[COUNT]] to i64
! CHECK: alloca {{.*}}, i64 %[[TOTAL]]
! CHECK-NOT: sdiv
! CHECK: %[[EMPTY:.*]] = icmp slt i32 %[[M]], 1
! CHECK: br i1 %[[EMPTY]], label %[[EXIT:.*]], label %[[PRE:.*]]
! CHECK: [[PRE]]:
! CHECK: %[[D:.*]] = load i32,
! CHECK: br label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32 %{{.*}}, %[[D]]
! CHECK: [[EXIT]]:
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 %[[COUNT]],

! An unused j must not mask an empty retained i.
subroutine depend_retained_empty(a, m, n)
  integer :: a(8), m, n
  !$omp task depend(iterator(i = 3:m, j = 1:n), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_retained_empty_(
! CHECK: %[[EMPTY:.*]] = icmp slt i32 %{{.*}}, 3
! CHECK: %[[TRIPS:.*]] = select i1 %[[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: %[[SIZE:.*]] = mul{{.*}} i64 %[[TRIPS]], {{[0-9]+}}
! CHECK: call ptr @malloc(i64 %[[SIZE]])
! CHECK: %[[ZERO:.*]] = icmp eq i64 %[[TRIPS]], 0
! CHECK: br i1 %[[ZERO]], label %[[EXIT:.*]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: [[EXIT]]:
! CHECK: %[[COUNT:.*]] = trunc i64 %[[TRIPS]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 %[[COUNT]],

! Ranges keep the iterator's kind; narrowing these bounds would empty the range.
subroutine depend_wide_kind(a)
  integer :: a(3)
  integer(16) :: lb, ub
  lb = 9223372036854775807_16
  ub = 9223372036854775809_16
  !$omp task depend(iterator(integer(16) :: i = lb:ub), in: a(i-lb+1))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_kind_(
! CHECK: %[[IV:.*]] = phi i64 [ 0, %entry ]
! CHECK: getelementptr {{.*}}[4 x i8], ptr %{{[0-9]+}}, i64 %[[IV]]
! CHECK: icmp eq i64 %{{.*}}, 3
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 3,

! The step is wider than the iterator's kind; narrowing it would empty {-128}.
subroutine depend_wide_step(a)
  integer :: a(-200:200)
  !$omp task depend(iterator(integer(1) :: i = -128:-1:200), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_step_(
! CHECK: getelementptr {{.*}} ptr %{{[0-9]+}}, i64 288
! CHECK: call i32 @__kmpc_omp_task_with_deps(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 1,

! Begin and end are not converted to the iterator's kind, so n = 256 leaves
! this range empty instead of wrapping to 0:0:-1.
subroutine depend_wide_end(a, n)
  integer :: a(8), n
  !$omp task depend(iterator(integer(1) :: i = 0:n:-1), in: a(1+8/i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_end_(
! CHECK: %[[N:.*]] = load i32, ptr
! CHECK: %[[EMPTY:.*]] = icmp sgt i32 %[[N]], 0
! CHECK: %[[COUNT:.*]] = select i1 %[[EMPTY]], i64 0, i64 %{{.*}}
! CHECK: %[[ZERO:.*]] = icmp eq i64 %[[COUNT]], 0
! CHECK: br i1 %[[ZERO]], label %[[CONT:[^,]+]], label %[[BODY:.*]]
! CHECK: [[BODY]]:
! CHECK: sdiv i32 8,
! CHECK: [[CONT]]:
! CHECK: %[[RUNTIME_COUNT:.*]] = trunc i64 %[[COUNT]] to i32
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 %[[RUNTIME_COUNT]],

subroutine depend_wide_end_const(a)
  integer :: a(8)
  !$omp task depend(iterator(integer(1) :: i = 0:256:-1), in: a(1+8/i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_end_const_(
! CHECK-NOT: sdiv
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 0,

subroutine depend_wide_begin_const(a)
  integer :: a(8)
  !$omp task depend(iterator(integer(1) :: i = 200:100), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_begin_const_(
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 0,

! Narrowing 150 to integer(1) would empty this range; its values are {0, 100}.
subroutine depend_wide_end_nonempty(a)
  integer :: a(0:100)
  !$omp task depend(iterator(integer(1) :: i = 0:150:100), in: a(i))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @depend_wide_end_nonempty_(
! CHECK: call i32 @__kmpc_omp_task_with_deps(ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 2,

subroutine affinity_wide_kind(a)
  integer :: a(3)
  integer(16) :: lb, ub
  lb = 9223372036854775807_16
  ub = 9223372036854775809_16
  !$omp task affinity(iterator(integer(16) :: i = lb:ub): a(i-lb+1))
  !$omp end task
end subroutine

! CHECK-LABEL: define {{.*}} @affinity_wide_kind_(
! CHECK: call i32 @__kmpc_omp_reg_task_with_affinity(
! CHECK-SAME: ptr {{[^,]+}}, i32 {{[^,]+}}, ptr {{[^,]+}}, i32 3,
