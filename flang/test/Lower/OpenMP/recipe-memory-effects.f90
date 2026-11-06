! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %s -o - | FileCheck %s

! Each clause has its own bounded-recipe-effects assumption, which requires
! all variables in that clause to be non-derived and non-allocatable.
module recipe_memory_effects
  implicit none
  type :: record
    sequence
    integer :: value
  end type
contains
  ! Intrinsic private and reduction variables are eligible independently.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_clauses(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <private_assume_bounded_recipe_effects, reduction_assume_bounded_recipe_effects>
  subroutine intrinsic_clauses(x, y, p, f, l)
    integer :: x, p, f, l
    real :: y
    !$omp sections reduction(+:x,y) private(p) firstprivate(f) lastprivate(l)
      p = f
      l = p
      x = x + p
      y = y + 1.0
    !$omp end sections
  end subroutine

  ! Non-derived private, firstprivate and lastprivate variables are eligible
  ! without a reduction clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_private(
  ! CHECK: omp.sections {
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine intrinsic_private(p, f, l)
    integer :: p, f, l
    !$omp sections private(p) firstprivate(f) lastprivate(l)
      p = f
      l = p
    !$omp end sections
  end subroutine

  ! An allocatable variable prevents the assumption for the whole private clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPboxed_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine boxed_private(a, p)
    real, allocatable :: a(:)
    real, pointer :: p
    !$omp sections private(a, p)
      a = 1.0
      nullify(p)
    !$omp end sections
  end subroutine

  ! Scalar allocatables are also excluded, independently of the other clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_private_scalar_reduction(
  ! CHECK: omp.sections reduction(
  ! CHECK-NOT: private_assume_bounded_recipe_effects
  ! CHECK: } <reduction_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine allocatable_private_scalar_reduction(a, x)
    real, allocatable :: a
    integer :: x
    !$omp sections private(a) reduction(+:x)
      x = x + 1
    !$omp end sections
  end subroutine

  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_reduction_scalar_private(
  ! CHECK: omp.sections reduction(
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine allocatable_reduction_scalar_private(a, x)
    real, allocatable :: a
    integer :: x
    !$omp sections reduction(+:a) private(x)
      x = 1
      a = a + x
    !$omp end sections
  end subroutine

  ! An assumed-shape array reduction uses a referenced descriptor. It prevents
  ! the assumption for the reduction clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsections_array_descriptor_reduction(
  ! CHECK: omp.sections nowait reduction(
  ! CHECK-SAME: : !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.ref<f32>) {
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine sections_array_descriptor_reduction(a, x)
    real :: a(:), x
    !$omp sections reduction(+:a,x)
      a = a + 1
      x = x + 2
    !$omp end sections nowait
  end subroutine

  ! Eager sections privatization creates character and array storage directly,
  ! including when the enclosing parallel uses a referenced array descriptor.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsections_character_array_private(
  ! CHECK: omp.parallel private(
  ! CHECK-SAME: : !fir.boxchar<1>, !fir.ref<!fir.box<!fir.array<?xf32>>>) {
  ! CHECK: omp.sections nowait reduction(
  ! CHECK-SAME: : !fir.ref<f32>) {
  ! CHECK: } <private_assume_bounded_recipe_effects, reduction_assume_bounded_recipe_effects>{{$}}
  ! CHECK: return
  subroutine sections_character_array_private(c, a, x)
    character(*) :: c
    real :: a(:), x
    !$omp parallel private(c,a)
      !$omp sections private(c,a) reduction(+:x)
        c = 'x'
        a = 1.0
        x = x + a(1)
      !$omp end sections nowait
    !$omp end parallel
  end subroutine

  ! A derived private variable prevents the assumption for the private clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPderived_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine derived_private()
    integer :: i
    type(record) :: z
    !$omp sections private(i, z)
      i = 1
      z%value = i
    !$omp end sections
  end subroutine

  ! The generated conditional-lastprivate record is eligible.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate(
  ! CHECK: omp.sections reduction(
  ! CHECK: } <reduction_assume_bounded_recipe_effects>
  subroutine conditional_lastprivate(x, y)
    integer :: x, y
    !$omp sections lastprivate(conditional:x) firstprivate(x) reduction(+:y)
      x = x + 1
      y = y + x
    !$omp end sections
  end subroutine

  ! A derived reduction prevents the assumption for the whole reduction clause,
  ! independently of the non-derived private variables.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPconditional_lastprivate_derived_reduction(
  ! CHECK: omp.sections reduction({{(byref )?}}@
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK-NOT: reduction_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine conditional_lastprivate_derived_reduction(x, z)
    integer :: x, p
    type(record) :: z
    !$omp declare reduction(record_sum : record : &
    !$omp & omp_out%value = omp_out%value + omp_in%value) &
    !$omp & initializer(omp_priv%value = 0)
    !$omp parallel
      !$omp sections lastprivate(conditional:x) reduction(record_sum:z) private(p)
        p = 1
        x = p
        z%value = z%value + x
      !$omp end sections
    !$omp end parallel
  end subroutine

  ! A common block is eligible when all its members are non-derived.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_common_private(
  ! CHECK: omp.sections {
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine intrinsic_common_private()
    integer :: x
    real :: y
    common /intrinsic_block/ x, y
    !$omp sections private(/intrinsic_block/)
      x = 1
      y = 1.0
    !$omp end sections
  end subroutine

  ! A derived common-block member prevents the private-clause assumption.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPderived_common_private(
  ! CHECK: omp.sections {
  ! CHECK-NOT: assume_bounded_recipe_effects
  ! CHECK: return
  subroutine derived_common_private()
    integer :: x
    type(record) :: z
    common /derived_block/ x, z
    !$omp sections private(/derived_block/)
      x = 1
      z%value = x
    !$omp end sections
  end subroutine

  ! Eager privatisation emits the privatisation recipe inside the single
  ! region, preventing getEffects() from providing its memory effects.
  ! Make sure these are emitted so recursive analyis records their memory
  ! effects.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsingle_eager_firstprivate(
  ! CHECK: %[[ORIGINAL:.*]] = fir.alloca i32
  ! CHECK: %[[ORIGINAL_DECL:.*]]:2 = hlfir.declare %[[ORIGINAL]]
  ! CHECK: omp.parallel {
  ! CHECK-NEXT: omp.single {
  ! CHECK-NEXT: %[[PRIVATE:.*]] = fir.alloca i32
  ! CHECK-NEXT: %[[PRIVATE_DECL:.*]]:2 = hlfir.declare %[[PRIVATE]]
  ! CHECK-NEXT: %[[INITIAL:.*]] = fir.load %[[ORIGINAL_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: hlfir.assign %[[INITIAL]] to %[[PRIVATE_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK-NEXT: %[[VALUE:.*]] = fir.load %[[PRIVATE_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
  ! CHECK-NEXT: %[[UPDATED:.*]] = arith.addi %[[VALUE]], %[[ONE]] : i32
  ! CHECK-NEXT: hlfir.assign %[[UPDATED]] to %[[PRIVATE_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: } <copyprivate_assume_bounded_recipe_effects, private_assume_bounded_recipe_effects>
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NEXT: }
  ! CHECK-NEXT: %[[AFTER:.*]] = fir.load %[[ORIGINAL_DECL]]#0 : !fir.ref<i32>
  ! CHECK-NEXT: hlfir.assign %[[AFTER]] to %{{.*}} : i32, !fir.ref<i32>
  ! CHECK-NEXT: return
  subroutine single_eager_firstprivate(result)
    integer :: result, x
    x = 10
    !$omp parallel shared(x)
      !$omp single firstprivate(x)
        x = x + 1
      !$omp end single
    !$omp end parallel
    result = x
  end subroutine

  ! Eager single privatization maps COMMON members rather than the block.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsingle_intrinsic_common_private(
  ! CHECK: omp.single {
  ! CHECK: } <copyprivate_assume_bounded_recipe_effects, private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine single_intrinsic_common_private()
    integer :: x
    real :: y
    common /single_private_block/ x, y
    !$omp single private(/single_private_block/)
      x = 1
      y = 1.0
    !$omp end single
  end subroutine

  ! Firstprivate COMMON blocks must also be checked member by member.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsingle_intrinsic_common_firstprivate(
  ! CHECK: omp.single {
  ! CHECK: } <copyprivate_assume_bounded_recipe_effects, private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine single_intrinsic_common_firstprivate()
    integer :: x
    real :: y
    common /single_firstprivate_block/ x, y
    !$omp single firstprivate(/single_firstprivate_block/)
      x = x + 1
      y = y + 1.0
    !$omp end single
  end subroutine

  ! One derived member prevents the assumption for the entire COMMON block.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPsingle_derived_common_private(
  ! CHECK: omp.single {
  ! CHECK-NOT: private_assume_bounded_recipe_effects
  ! CHECK: } <copyprivate_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine single_derived_common_private()
    integer :: x
    type(record) :: z
    common /single_derived_block/ x, z
    !$omp single private(/single_derived_block/)
      x = 1
      z%value = x
    !$omp end single
  end subroutine

  ! Direct intrinsic scalars and fixed-size arrays have bounded copy functions.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_copyprivate(
  ! CHECK: omp.single copyprivate(
  ! CHECK: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine intrinsic_copyprivate()
    integer :: i, a(4)
    real :: r
    complex :: c
    logical :: l
    character(8) :: s
    !$omp parallel private(i, a, r, c, l, s)
      !$omp single
      !$omp end single copyprivate(i, a, r, c, l, s)
    !$omp end parallel
  end subroutine

  ! Copying through a referenced descriptor accesses its pointee even when
  ! the Fortran variable is neither allocatable nor a pointer.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPcharacter_descriptor_copyprivate(
  ! CHECK: omp.single copyprivate({{.*}} : !fir.ref<!fir.boxchar<1>>) {
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine character_descriptor_copyprivate(c)
    character(*) :: c
    !$omp parallel private(c)
      !$omp single
        c = 'x'
      !$omp end single copyprivate(c)
    !$omp end parallel
  end subroutine

  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsParray_descriptor_copyprivate(
  ! CHECK: omp.single copyprivate({{.*}} : !fir.ref<!fir.box<!fir.array<?xf32>>>) {
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine array_descriptor_copyprivate(a)
    real :: a(:)
    !$omp parallel private(a)
      !$omp single
        a = 1.0
      !$omp end single copyprivate(a)
    !$omp end parallel
  end subroutine

  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPno_copyprivate(
  ! CHECK: omp.single {
  ! CHECK: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine no_copyprivate()
    !$omp single
    !$omp end single
  end subroutine

  ! Copyprivate and private make independent assumptions.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPbounded_copyprivate_derived_private(
  ! CHECK: omp.single copyprivate(
  ! CHECK: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine bounded_copyprivate_derived_private()
    type record
      integer :: value
    end type
    type(record) :: z
    integer :: i
    !$omp parallel private(i)
      !$omp single private(z)
      !$omp end single copyprivate(i)
    !$omp end parallel
  end subroutine

  ! One derived item prevents the assumption for the entire copyprivate clause.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPderived_copyprivate_bounded_private(
  ! CHECK: omp.single copyprivate(
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: } <private_assume_bounded_recipe_effects>
  ! CHECK: return
  subroutine derived_copyprivate_bounded_private()
    type record
      integer :: value
    end type
    type(record) :: z
    integer :: i, p
    !$omp parallel private(i, z)
      !$omp single private(p)
      !$omp end single copyprivate(i, z)
    !$omp end parallel
  end subroutine

  ! Allocatable copies may allocate, free, and access pointee storage.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_scalar_copyprivate(
  ! CHECK: omp.single copyprivate(
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine allocatable_scalar_copyprivate()
    integer, allocatable :: a
    !$omp parallel private(a)
      !$omp single
      !$omp end single copyprivate(a)
    !$omp end parallel
  end subroutine

  ! Allocatables may access pointees of the copyprivate variable, violating the
  ! copyprivate_assume_bounded_recipe_effects attribute contract.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPallocatable_array_copyprivate(
  ! CHECK: omp.single copyprivate(
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine allocatable_array_copyprivate()
    integer, allocatable :: a(:)
    !$omp parallel private(a)
      !$omp single
      !$omp end single copyprivate(a)
    !$omp end parallel
  end subroutine

  ! Pointer variables may access pointees of the copyprivate variable, violating
  ! the copyprivate_assume_bounded_recipe_effects attribute contract.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPpointer_copyprivate(
  ! CHECK: omp.single copyprivate(
  ! CHECK-NOT: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine pointer_copyprivate()
    integer, pointer :: p, a(:)
    !$omp parallel private(p, a)
      !$omp single
      !$omp end single copyprivate(p, a)
    !$omp end parallel
  end subroutine

  ! Check the members of a threadprivate COMMON block.
  ! CHECK-LABEL: func.func @_QMrecipe_memory_effectsPintrinsic_common_copyprivate(
  ! CHECK: omp.single copyprivate(
  ! CHECK: copyprivate_assume_bounded_recipe_effects
  ! CHECK: return
  subroutine intrinsic_common_copyprivate()
    integer :: i
    real :: r
    common /intrinsic_copy_block/ i, r
    !$omp threadprivate(/intrinsic_copy_block/)
    !$omp parallel
      !$omp single
      !$omp end single copyprivate(/intrinsic_copy_block/)
    !$omp end parallel
  end subroutine
end module
