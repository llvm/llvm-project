! Test lowering of metadirectives with ordinary loop-associated variants.

! RUN: %flang_fc1 -fopenmp -emit-hlfir -fopenmp-version=52 %s -o - | \
! RUN:   FileCheck %s --implicit-check-not=omp.parallel \
! RUN:     --implicit-check-not=omp.canonical_loop

! CHECK: #[[UNROLL:loop_unroll[0-9]*]] =
! CHECK-SAME: #llvm.loop_unroll<disable = false, count = 4 : i64>
! CHECK: #[[VECTORIZE:loop_vectorize[0-9]*]] =
! CHECK-SAME: #llvm.loop_vectorize<disable = false>
! CHECK: #[[UNROLL_ANNOTATION:loop_annotation[0-9]*]] =
! CHECK-SAME: #llvm.loop_annotation<unroll = #[[UNROLL]]>
! CHECK: #[[VECTOR_ANNOTATION:loop_annotation[0-9]*]] =
! CHECK-SAME: #llvm.loop_annotation<vectorize = #[[VECTORIZE]]>

! CHECK-LABEL: func.func @_QPtest_do(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[DO_I:.*]]:2 = hlfir.declare {{.*}}_QFtest_doEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop private(@_QFtest_doEi_private_i32
! CHECK-SAME:      %[[DO_I]]#0 ->
! CHECK-SAME:      %[[DO_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[DO_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[DO_PRIVATE_DECL:.*]]:2 =
! CHECK-SAME:          hlfir.declare %[[DO_PRIVATE]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             hlfir.assign %[[DO_IV]] to %[[DO_PRIVATE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[DO_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[DO_PRIVATE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[DO_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[DO_PRIVATE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[DO_INDEX_I64:.*]] = fir.convert %[[DO_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[DO_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:          (%[[DO_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign %[[DO_VALUE]] to %[[DO_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_do(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_simd(
! CHECK:         %[[SIMD_AFTER:.*]]:2 = hlfir.declare %arg2
! CHECK-SAME:      uniq_name = "_QFtest_simdEafter"
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[SIMD_I:.*]]:2 = hlfir.declare {{.*}}_QFtest_simdEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.simd linear(
! CHECK-SAME:      val(%[[SIMD_I]]#0 : !fir.ref<i32> =
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[SIMD_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             hlfir.assign %[[SIMD_IV]] to %[[SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[SIMD_VALUE:.*]] = fir.load %[[SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[SIMD_INDEX:.*]] = fir.load %[[SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[SIMD_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[SIMD_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SIMD_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:          (%[[SIMD_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign %[[SIMD_VALUE]] to %[[SIMD_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         %[[SIMD_AFTER_VALUE:.*]] = fir.load %[[SIMD_I]]#0
! CHECK:         hlfir.assign %[[SIMD_AFTER_VALUE]] to %[[SIMD_AFTER]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_simd(n, a, after)
  integer :: n, a(n), after, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
end subroutine

! CHECK-LABEL: func.func @_QPtest_do_simd(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[DO_SIMD_I:.*]]:2 = hlfir.declare {{.*}}_QFtest_do_simdEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.simd linear(
! CHECK-SAME:        val(%[[DO_SIMD_I]]#0 : !fir.ref<i32> =
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[DO_SIMD_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               hlfir.assign %[[DO_SIMD_IV]] to
! CHECK-SAME:            %[[DO_SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DO_SIMD_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[DO_SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DO_SIMD_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[DO_SIMD_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DO_SIMD_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[DO_SIMD_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DO_SIMD_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[DO_SIMD_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[DO_SIMD_VALUE]] to
! CHECK-SAME:            %[[DO_SIMD_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_do_simd(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_begin_do(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[BEGIN_DO_I:.*]]:2 = hlfir.declare {{.*}}_QFtest_begin_doEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private(@_QFtest_begin_doEi_private_i32
! CHECK-SAME:        %[[BEGIN_DO_I]]#0 ->
! CHECK-SAME:        %[[BEGIN_DO_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[BEGIN_DO_IV:.*]]) :
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[BEGIN_DO_PRIVATE_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[BEGIN_DO_PRIVATE]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               hlfir.assign %[[BEGIN_DO_IV]] to
! CHECK-SAME:            %[[BEGIN_DO_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[BEGIN_DO_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[BEGIN_DO_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[BEGIN_DO_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[BEGIN_DO_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[BEGIN_DO_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[BEGIN_DO_INDEX]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[BEGIN_DO_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[BEGIN_DO_INDEX_I64]])
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[BEGIN_DO_VALUE]] to
! CHECK-SAME:            %[[BEGIN_DO_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop %[[BEGIN_DO_FALLBACK_IV:.*]] =
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             fir.store %[[BEGIN_DO_FALLBACK_IV]] to
! CHECK-SAME:          %[[BEGIN_DO_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[BEGIN_DO_FALLBACK_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[BEGIN_DO_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[BEGIN_DO_FALLBACK_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[BEGIN_DO_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[BEGIN_DO_FALLBACK_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[BEGIN_DO_FALLBACK_INDEX]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[BEGIN_DO_FALLBACK_ELEMENT:.*]] =
! CHECK-SAME:          hlfir.designate
! CHECK-SAME:          (%[[BEGIN_DO_FALLBACK_INDEX_I64]])
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             hlfir.assign %[[BEGIN_DO_FALLBACK_VALUE]] to
! CHECK-SAME:          %[[BEGIN_DO_FALLBACK_ELEMENT]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.store {{.*}} to %[[BEGIN_DO_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK-NOT:     omp.
! CHECK:         return
subroutine test_begin_do(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! The following loop must remain available when the PFT is reused for ENTRY.
! CHECK-LABEL: func.func @_QPtest_standalone_entry_no_directive(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[ENTRY_I:.*]]:2 = hlfir.declare {{.*}}no_directiveEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[ENTRY_I]]#0 ->
! CHECK-SAME:        %[[ENTRY_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[ENTRY_IV:.*]]) :
! CHECK:               %[[ENTRY_PRIVATE_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[ENTRY_PRIVATE]]
! CHECK:               hlfir.assign %[[ENTRY_IV]] to
! CHECK-SAME:            %[[ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ENTRY_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ENTRY_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ENTRY_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[ENTRY_INDEX]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               %[[ENTRY_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[ENTRY_INDEX_I64]])
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               hlfir.assign %[[ENTRY_VALUE]] to %[[ENTRY_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop %[[ENTRY_FALLBACK_IV:.*]] =
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             fir.store %[[ENTRY_FALLBACK_IV]] to
! CHECK-SAME:          %[[ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ENTRY_FALLBACK_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ENTRY_FALLBACK_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ENTRY_FALLBACK_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[ENTRY_FALLBACK_INDEX]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ENTRY_FALLBACK_ELEMENT:.*]] =
! CHECK-SAME:          hlfir.designate
! CHECK-SAME:          (%[[ENTRY_FALLBACK_INDEX_I64]])
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             hlfir.assign %[[ENTRY_FALLBACK_VALUE]] to
! CHECK-SAME:          %[[ENTRY_FALLBACK_ELEMENT]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.store {{.*}} to %[[ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK-NOT:     omp.
! CHECK:         return
! CHECK-LABEL: func.func @_QPtest_alt_standalone_entry_no_directive(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[ALT_ENTRY_I:.*]]:2 = hlfir.declare {{.*}}no_directiveEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[ALT_ENTRY_I]]#0 ->
! CHECK-SAME:        %[[ALT_ENTRY_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[ALT_ENTRY_IV:.*]]) :
! CHECK:               %[[ALT_ENTRY_PRIVATE_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[ALT_ENTRY_PRIVATE]]
! CHECK:               hlfir.assign %[[ALT_ENTRY_IV]] to
! CHECK-SAME:            %[[ALT_ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ALT_ENTRY_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[ALT_ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ALT_ENTRY_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[ALT_ENTRY_PRIVATE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[ALT_ENTRY_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[ALT_ENTRY_INDEX]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               %[[ALT_ENTRY_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[ALT_ENTRY_INDEX_I64]])
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               hlfir.assign %[[ALT_ENTRY_VALUE]] to
! CHECK-SAME:            %[[ALT_ENTRY_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop %[[ALT_ENTRY_FALLBACK_IV:.*]] =
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             fir.store %[[ALT_ENTRY_FALLBACK_IV]] to
! CHECK-SAME:          %[[ALT_ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ALT_ENTRY_FALLBACK_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[ALT_ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ALT_ENTRY_FALLBACK_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[ALT_ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ALT_ENTRY_FALLBACK_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[ALT_ENTRY_FALLBACK_INDEX]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[ALT_ENTRY_FALLBACK_ELEMENT:.*]] =
! CHECK-SAME:          hlfir.designate
! CHECK-SAME:          (%[[ALT_ENTRY_FALLBACK_INDEX_I64]])
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             hlfir.assign %[[ALT_ENTRY_FALLBACK_VALUE]] to
! CHECK-SAME:          %[[ALT_ENTRY_FALLBACK_ELEMENT]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.store {{.*}} to %[[ALT_ENTRY_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:         }
! CHECK-NOT:     fir.do_loop
! CHECK-NOT:     omp.
! CHECK:         return
! CHECK-LABEL: func.func @_QPtest_after_standalone_entry_no_directive(
! CHECK-NOT:     fir.if
! CHECK-NOT:     omp.
! CHECK-NOT:     fir.do_loop
! CHECK:         %[[AFTER_ENTRY_C77:.*]] = arith.constant 77 : i32
! CHECK:         hlfir.assign %[[AFTER_ENTRY_C77]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_standalone_entry_no_directive(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  entry test_alt_standalone_entry_no_directive(flag, n, a)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  entry test_after_standalone_entry_no_directive(n, a)
  a(1) = 77
end subroutine

! A statically inapplicable loop variant nested in a parallel region leaves the
! following loop sequential.
! CHECK-LABEL: func.func @_QPtest_inapplicable_do_in_parallel(
! CHECK:         omp.parallel
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:           fir.do_loop
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[INAPPLICABLE_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign {{.*}} to %[[INAPPLICABLE_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.terminator
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_inapplicable_do_in_parallel(n, a, after)
  integer :: n, a(n), after, i
  !$omp parallel num_threads(1) shared(n, a, after)
  !$omp metadirective &
  !$omp & when(implementation={vendor("unknown")}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

! An unreachable loop variant likewise does not turn a statically selected
! block variant into a mixed-association metadirective.
! CHECK-LABEL: func.func @_QPtest_unselected_do_with_block_variant(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         omp.masked
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:           fir.do_loop
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[UNSELECTED_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign {{.*}} to %[[UNSELECTED_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.terminator
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_unselected_do_with_block_variant(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(score(2): .true.)}: masked) &
  !$omp & when(user={condition(score(1): .true.)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! A lower-ranked candidate guarded by the same runtime expression is
! unreachable: when FLAG is true the higher-ranked BARRIER wins, and when it
! is false neither guarded candidate matches. Do not emit a dead OpenMP loop.
! CHECK-LABEL: func.func @_QPtest_unreachable_same_runtime_condition(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:           %[[UNREACHABLE_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:           hlfir.assign {{.*}} to %[[UNREACHABLE_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_unreachable_same_runtime_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Parentheses do not make a repeatable condition distinct. The lower-ranked
! loop remains unreachable and must not be emitted.
! CHECK-LABEL: func.func @_QPtest_unreachable_parenthesized_condition(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_unreachable_parenthesized_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): (flag))}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Idempotent AND/OR spelling is normalized after proving that the condition is
! repeatable.
! CHECK-LABEL: func.func @_QPtest_unreachable_idempotent_condition(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         fir.if {{.*}} {
! CHECK:           omp.barrier
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)}}
! CHECK:         }
! CHECK:         fir.do_loop
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)}}
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_unreachable_idempotent_condition(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): flag)}: barrier) &
  !$omp & when(user={condition(score(1): flag .or. flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Calls to an opaque procedure are independent runtime conditions even when
! their source expressions are identical. Preserve both candidates without
! relying on clause-expression side effects.
! CHECK-LABEL: func.func @_QPtest_opaque_runtime_conditions(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[OPAQUE_I:.*]]:2 = hlfir.declare {{.*}}conditionsEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.call @_QPmetadirective_runtime_condition
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.barrier
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.do_loop %[[OPAQUE_HIGH_IV:.*]] =
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             fir.store %[[OPAQUE_HIGH_IV]] to %[[OPAQUE_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[OPAQUE_HIGH_VALUE:.*]] = fir.load %[[OPAQUE_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[OPAQUE_HIGH_INDEX:.*]] = fir.load %[[OPAQUE_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[OPAQUE_HIGH_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign %[[OPAQUE_HIGH_VALUE]] to
! CHECK-SAME:          %[[OPAQUE_HIGH_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.store {{.*}} to %[[OPAQUE_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.call @_QPmetadirective_runtime_condition
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.if {{.*}} {
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:          %[[OPAQUE_I]]#0 ->
! CHECK-SAME:          %[[OPAQUE_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               omp.loop_nest (%[[OPAQUE_OMP_IV:.*]]) :
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:                 %[[OPAQUE_DECL:.*]]:2 =
! CHECK-SAME:              hlfir.declare %[[OPAQUE_PRIVATE]]
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:                 hlfir.assign %[[OPAQUE_OMP_IV]] to
! CHECK-SAME:              %[[OPAQUE_DECL]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[OPAQUE_OMP_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign {{.*}} to %[[OPAQUE_OMP_ELEMENT]]
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 omp.yield
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           } else {
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             fir.do_loop %[[OPAQUE_LOW_IV:.*]] =
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               fir.store %[[OPAQUE_LOW_IV]] to %[[OPAQUE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[OPAQUE_LOW_VALUE:.*]] = fir.load %[[OPAQUE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[OPAQUE_LOW_INDEX:.*]] = fir.load %[[OPAQUE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[OPAQUE_LOW_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[OPAQUE_LOW_VALUE]] to
! CHECK-SAME:            %[[OPAQUE_LOW_ELEMENT]]
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:             }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             fir.store {{.*}} to %[[OPAQUE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           }
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_opaque_runtime_conditions(n, a)
  integer :: n, a(n), i
  logical :: metadirective_runtime_condition
  external :: metadirective_runtime_condition
  !$omp metadirective &
  !$omp & when(user={condition(score(2): metadirective_runtime_condition())}: barrier) &
  !$omp & when(user={condition(score(1): metadirective_runtime_condition())}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

module metadirective_condition_helpers
contains
  pure logical function metadirective_identity(value)
    logical, intent(in) :: value
    metadirective_identity = value
  end function
end module

! Procedure calls are conservatively kept as independent runtime conditions
! because the expression tree does not describe the callee's state.
! CHECK-LABEL: func.func @_QPtest_pure_runtime_conditions(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[PURE_I:.*]]:2 = hlfir.declare {{.*}}conditionsEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.call {{.*}}Pmetadirective_identity
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[PURE_I]]#0 ->
! CHECK-SAME:        %[[PURE_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[PURE_DO_IV:.*]]) :
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[PURE_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[PURE_PRIVATE]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               hlfir.assign %[[PURE_DO_IV]] to %[[PURE_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[PURE_DO_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign {{.*}} to %[[PURE_DO_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.call {{.*}}Pmetadirective_identity
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.if {{.*}} {
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.simd linear(
! CHECK-SAME:          val(%[[PURE_I]]#0 : !fir.ref<i32> =
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               omp.loop_nest (%[[PURE_SIMD_IV:.*]]) :
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:                 hlfir.assign %[[PURE_SIMD_IV]] to %[[PURE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[PURE_SIMD_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign {{.*}} to %[[PURE_SIMD_ELEMENT]]
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 omp.yield
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           } else {
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             fir.do_loop %[[PURE_FB_IV:.*]] =
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               fir.store %[[PURE_FB_IV]] to %[[PURE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[PURE_FB_VALUE:.*]] = fir.load %[[PURE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[PURE_FB_INDEX:.*]] = fir.load %[[PURE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[PURE_FB_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[PURE_FB_VALUE]] to
! CHECK-SAME:            %[[PURE_FB_ELEMENT]]
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:             }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             fir.store {{.*}} to %[[PURE_I]]#0
! CHECK-NOT:         {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           }
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_pure_runtime_conditions(flag, n, a)
  use metadirective_condition_helpers, only : metadirective_identity
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(score(2): &
  !$omp &   metadirective_identity(flag))}: do) &
  !$omp & when(user={condition(score(1): &
  !$omp &   metadirective_identity(flag))}: simd) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_dynamic_loop(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[DYNAMIC_I:.*]]:2 = hlfir.declare {{.*}}_QFtest_dynamic_loopEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[DYNAMIC_I]]#0 ->
! CHECK-SAME:        %[[DYNAMIC_DO_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[DYNAMIC_DO_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_DO_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[DYNAMIC_DO_PRIVATE]]
! CHECK:               hlfir.assign %[[DYNAMIC_DO_IV]] to
! CHECK-SAME:            %[[DYNAMIC_DO_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_DO_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[DYNAMIC_DO_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_DO_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[DYNAMIC_DO_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_DO_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[DYNAMIC_DO_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DYNAMIC_DO_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[DYNAMIC_DO_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[DYNAMIC_DO_VALUE]] to
! CHECK-SAME:            %[[DYNAMIC_DO_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.simd linear(
! CHECK-SAME:        val(%[[DYNAMIC_I]]#0 : !fir.ref<i32> =
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[DYNAMIC_SIMD_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               hlfir.assign %[[DYNAMIC_SIMD_IV]] to
! CHECK-SAME:            %[[DYNAMIC_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_SIMD_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[DYNAMIC_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_SIMD_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[DYNAMIC_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DYNAMIC_SIMD_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[DYNAMIC_SIMD_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DYNAMIC_SIMD_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[DYNAMIC_SIMD_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[DYNAMIC_SIMD_VALUE]] to
! CHECK-SAME:            %[[DYNAMIC_SIMD_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_loop(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(simd)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! When the standalone fallback is selected at runtime, the following loop is
! lowered sequentially in that arm.
! CHECK-LABEL: func.func @_QPtest_dynamic_standalone_fallback(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[STANDALONE_I:.*]]:2 = hlfir.declare {{.*}}fallbackEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[STANDALONE_I]]#0 ->
! CHECK-SAME:        %[[STANDALONE_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[STANDALONE_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[STANDALONE_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[STANDALONE_PRIVATE]]
! CHECK:               hlfir.assign %[[STANDALONE_IV]] to
! CHECK-SAME:            %[[STANDALONE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[STANDALONE_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[STANDALONE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[STANDALONE_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[STANDALONE_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[STANDALONE_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[STANDALONE_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[STANDALONE_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[STANDALONE_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[STANDALONE_VALUE]] to
! CHECK-SAME:            %[[STANDALONE_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.barrier
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.do_loop %[[STANDALONE_FB_IV:.*]] =
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             fir.store %[[STANDALONE_FB_IV]] to
! CHECK-SAME:          %[[STANDALONE_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[STANDALONE_FB_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[STANDALONE_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[STANDALONE_FB_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[STANDALONE_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[STANDALONE_FB_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[STANDALONE_FB_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[STANDALONE_FB_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:          (%[[STANDALONE_FB_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:       {{^ *}}}
! CHECK:             hlfir.assign %[[STANDALONE_FB_VALUE]] to
! CHECK-SAME:          %[[STANDALONE_FB_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.store {{.*}} to %[[STANDALONE_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_standalone_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(barrier)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! When NOTHING is selected, the following loop is lowered normally.
! CHECK-LABEL: func.func @_QPtest_dynamic_nothing_fallback(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[NOTHING_I:.*]]:2 = hlfir.declare {{.*}}fallbackEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[NOTHING_I]]#0 ->
! CHECK-SAME:        %[[NOTHING_PRIVATE:.*]] : !fir.ref<i32>)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[NOTHING_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[NOTHING_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[NOTHING_PRIVATE]]
! CHECK:               hlfir.assign %[[NOTHING_IV]] to
! CHECK-SAME:            %[[NOTHING_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[NOTHING_VALUE:.*]] =
! CHECK-SAME:            fir.load %[[NOTHING_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[NOTHING_INDEX:.*]] =
! CHECK-SAME:            fir.load %[[NOTHING_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[NOTHING_INDEX_I64:.*]] =
! CHECK-SAME:            fir.convert %[[NOTHING_INDEX]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[NOTHING_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[NOTHING_INDEX_I64]])
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[NOTHING_VALUE]] to
! CHECK-SAME:            %[[NOTHING_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop %[[NOTHING_FB_IV:.*]] =
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             fir.store %[[NOTHING_FB_IV]] to
! CHECK-SAME:          %[[NOTHING_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[NOTHING_FB_VALUE:.*]] =
! CHECK-SAME:          fir.load %[[NOTHING_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[NOTHING_FB_INDEX:.*]] =
! CHECK-SAME:          fir.load %[[NOTHING_I]]#0
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[NOTHING_FB_INDEX_I64:.*]] =
! CHECK-SAME:          fir.convert %[[NOTHING_FB_INDEX]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             %[[NOTHING_FB_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:          (%[[NOTHING_FB_INDEX_I64]])
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:             hlfir.assign %[[NOTHING_FB_VALUE]] to
! CHECK-SAME:          %[[NOTHING_FB_ELEMENT]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.store {{.*}} to %[[NOTHING_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_nothing_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! Compiler directives preceding the associated loop are processed before it.
! CHECK-LABEL: func.func @_QPtest_dynamic_unroll_fallback(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop
! CHECK-SAME:        attributes {loopAnnotation = #[[UNROLL_ANNOTATION]]}
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_unroll_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  !dir$ unroll 4
  do i = 1, n
    a(i) = i
  end do
end subroutine

! A supported compiler directive nested inside a begin/end metadirective is
! attached to the associated loop before runtime selection.
! CHECK-LABEL: func.func @_QPtest_begin_unroll_fallback(
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               %[[BEGIN_UNROLL_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               hlfir.assign {{.*}} to %[[BEGIN_UNROLL_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:           fir.do_loop
! CHECK-SAME:        attributes {loopAnnotation = #[[UNROLL_ANNOTATION]]}
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[BEGIN_UNROLL_FALLBACK_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign {{.*}} to %[[BEGIN_UNROLL_FALLBACK_ELEMENT]]
! CHECK-NOT:       omp.
! CHECK-NOT:       fir.do_loop
! CHECK:         }
! CHECK-NOT:     omp.
! CHECK-NOT:     fir.do_loop
! CHECK:         return
subroutine test_begin_unroll_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  !dir$ unroll 4
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

! Other compiler directives that do not emit executable operations may also
! appear between the metadirective and its associated loop. Check a loop
! annotation, an inlining annotation, and an unrecognized no-op directive.
! CHECK-LABEL: func.func @_QPtest_dynamic_intervening_compiler_directives(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.wsloop
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:               fir.call @_QPconsume
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           fir.do_loop
! CHECK-SAME:        attributes {loopAnnotation = #[[VECTOR_ANNOTATION]]}
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             fir.call @_QPconsume
! CHECK-SAME:          inline_attr = #fir.inline_attrs<always_inline>
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_intervening_compiler_directives(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  external :: consume
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  !dir$ vector always
  !dir$ forceinline
  !dir$ unknown
  do i = 1, n
    call consume(a(i))
  end do
end subroutine

! Each runtime arm must compute its own affected depth and restore temporary
! loop-index attributes before lowering the next arm.
! CHECK-LABEL: func.func @_QPtest_dynamic_collapse(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[DC_N:.*]]:2 = hlfir.declare {{.*}}dynamic_collapseEn
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[DC_I:.*]]:2 = hlfir.declare {{.*}}dynamic_collapseEi
! CHECK: %[[DC_J:.*]]:2 = hlfir.declare {{.*}}dynamic_collapseEj
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         fir.if {{.*}} {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.simd private({{[^,]*}}Ei_private_i32
! CHECK-SAME:        %[[DC_I]]#0 -> %[[DC_I_PRIV:[^, ]+]],
! CHECK-SAME:        {{.*}}Ej_private_i32
! CHECK-SAME:        %[[DC_J]]#0 -> %[[DC_J_PRIV:[^ ]+]] :
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[DC_I_IV:[^, ]+]], %[[DC_J_IV:[^) ]+]])
! CHECK-SAME:          collapse(2)
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               %[[DC_I_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[DC_I_PRIV]]
! CHECK:               %[[DC_J_DECL:.*]]:2 =
! CHECK-SAME:            hlfir.declare %[[DC_J_PRIV]]
! CHECK:               hlfir.assign %[[DC_I_IV]] to %[[DC_I_DECL]]#0
! CHECK:               hlfir.assign %[[DC_J_IV]] to %[[DC_J_DECL]]#0
! CHECK:               %[[DC_I_VAL:.*]] = fir.load %[[DC_I_DECL]]#0
! CHECK:               %[[DC_J_VAL:.*]] = fir.load %[[DC_J_DECL]]#0
! CHECK:               %[[DC_SUM:.*]] = arith.addi
! CHECK-SAME:            %[[DC_I_VAL]], %[[DC_J_VAL]]
! CHECK:               %[[DC_J_IDX:.*]] = fir.load %[[DC_J_DECL]]#0
! CHECK:               %[[DC_J_I64:.*]] = fir.convert %[[DC_J_IDX]]
! CHECK:               %[[DC_I_IDX:.*]] = fir.load %[[DC_I_DECL]]#0
! CHECK:               %[[DC_I_I64:.*]] = fir.convert %[[DC_I_IDX]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[DC_J_I64]], %[[DC_I_I64]])
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[DC_SUM]] to %[[DC_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_I_BOUND:.*]] = fir.load %[[DC_N]]#0
! CHECK-NEXT:          %[[DC_I_STEP:.*]] = arith.constant 1 : i32
! CHECK-NEXT:          %[[DC_J_BOUND:.*]] = fir.load %[[DC_N]]#0
! CHECK-NEXT:          %[[DC_J_STEP:.*]] = arith.constant 1 : i32
! CHECK-NEXT:          %[[DC_I_NEXT:.*]] = arith.addi
! CHECK-SAME:            %[[DC_I_IV]], %[[DC_I_STEP]]
! CHECK-NEXT:          %[[DC_I_ZERO:.*]] = arith.constant 0 : i32
! CHECK-NEXT:          %[[DC_I_NEG:.*]] = arith.cmpi slt,
! CHECK-SAME:            %[[DC_I_STEP]], %[[DC_I_ZERO]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_I_LT:.*]] = arith.cmpi slt,
! CHECK-SAME:            %[[DC_I_NEXT]], %[[DC_I_BOUND]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_I_GT:.*]] = arith.cmpi sgt,
! CHECK-SAME:            %[[DC_I_NEXT]], %[[DC_I_BOUND]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_I_LAST:.*]] = arith.select
! CHECK-SAME:            %[[DC_I_NEG]], %[[DC_I_LT]], %[[DC_I_GT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_J_NEXT:.*]] = arith.addi
! CHECK-SAME:            %[[DC_J_IV]], %[[DC_J_STEP]]
! CHECK-NEXT:          %[[DC_J_ZERO:.*]] = arith.constant 0 : i32
! CHECK-NEXT:          %[[DC_J_NEG:.*]] = arith.cmpi slt,
! CHECK-SAME:            %[[DC_J_STEP]], %[[DC_J_ZERO]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_J_LT:.*]] = arith.cmpi slt,
! CHECK-SAME:            %[[DC_J_NEXT]], %[[DC_J_BOUND]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_J_GT:.*]] = arith.cmpi sgt,
! CHECK-SAME:            %[[DC_J_NEXT]], %[[DC_J_BOUND]] : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_J_LAST:.*]] = arith.select
! CHECK-SAME:            %[[DC_J_NEG]], %[[DC_J_LT]], %[[DC_J_GT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[DC_LAST:.*]] = arith.andi
! CHECK-SAME:            %[[DC_I_LAST]], %[[DC_J_LAST]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               fir.if %[[DC_LAST]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign %[[DC_I_NEXT]] to %[[DC_I_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign %[[DC_J_NEXT]] to %[[DC_J_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[DC_I_COPY:.*]] = fir.load %[[DC_I_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign %[[DC_I_COPY]] to %[[DC_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[DC_J_COPY:.*]] = fir.load %[[DC_J_DECL]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign %[[DC_J_COPY]] to %[[DC_J]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:               }
! CHECK-NEXT: omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         } else {
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.simd linear(
! CHECK-SAME:        val(%[[DC_I]]#0 : !fir.ref<i32> =
! CHECK-SAME:        %[[DC_FB_STEP:[^ ]+]] : i32))
! CHECK-SAME:        linear_var_types([i32]) {
! CHECK-NOT:         Ei_private
! CHECK-NOT:         Ej_private
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.loop_nest (%[[DC_FB_IV:.*]]) : i32
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[DC_FB_IV]] to %[[DC_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:               fir.do_loop %[[DC_INNER_IV:.*]] =
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 fir.store %[[DC_INNER_IV]] to %[[DC_J]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[DC_FB_I_VALUE:.*]] = fir.load %[[DC_I]]#0
! CHECK:                 %[[DC_FB_J_VALUE:.*]] = fir.load %[[DC_J]]#0
! CHECK:                 %[[DC_FB_SUM:.*]] = arith.addi
! CHECK-SAME:              %[[DC_FB_I_VALUE]], %[[DC_FB_J_VALUE]]
! CHECK:                 %[[DC_FB_J_INDEX:.*]] = fir.load %[[DC_J]]#0
! CHECK:                 %[[DC_FB_I_INDEX:.*]] = fir.load %[[DC_I]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 %[[DC_FB_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:                 hlfir.assign %[[DC_FB_SUM]] to
! CHECK-SAME:              %[[DC_FB_ELEMENT]]
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:             {{^ *}}}
! CHECK:               }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               fir.store {{.*}} to %[[DC_J]]#0
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.yield
! CHECK-NOT:       {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         }
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_dynamic_collapse(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: simd collapse(2)) &
  !$omp & otherwise(simd)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_ordered_depth(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[OD_I:.*]]:2 = hlfir.declare {{.*}}ordered_depthEi
! CHECK: %[[OD_J:.*]]:2 = hlfir.declare {{.*}}ordered_depthEj
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop ordered(2) private({{[^,]*}}Ei_private_i32
! CHECK-SAME:      %[[OD_I]]#0 -> %[[OD_I_PRIV:[^, ]+]],
! CHECK-SAME:      {{.*}}Ej_private_i32
! CHECK-SAME:      %[[OD_J]]#0 -> %[[OD_J_PRIV:[^ ]+]] :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[OD_I_IV:.*]]) : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[OD_I_DECL:.*]]:2 = hlfir.declare %[[OD_I_PRIV]]
! CHECK:             %[[OD_J_DECL:.*]]:2 = hlfir.declare %[[OD_J_PRIV]]
! CHECK:             hlfir.assign %[[OD_I_IV]] to %[[OD_I_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             fir.do_loop %[[OD_J_IV:.*]] =
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               fir.store %[[OD_J_IV]] to %[[OD_J_DECL]]#0
! CHECK-NEXT:          %[[OD_I_VALUE:.*]] = fir.load %[[OD_I_DECL]]#0
! CHECK-NEXT:          %[[OD_J_VALUE:.*]] = fir.load %[[OD_J_DECL]]#0
! CHECK-NEXT:          %[[OD_SUM:.*]] = arith.addi
! CHECK-SAME:            %[[OD_I_VALUE]], %[[OD_J_VALUE]]
! CHECK-NEXT:          %[[OD_J_INDEX:.*]] = fir.load %[[OD_J_DECL]]#0
! CHECK-NEXT:          %[[OD_J_I64:.*]] = fir.convert %[[OD_J_INDEX]]
! CHECK-NEXT:          %[[OD_I_INDEX:.*]] = fir.load %[[OD_I_DECL]]#0
! CHECK-NEXT:          %[[OD_I_I64:.*]] = fir.convert %[[OD_I_INDEX]]
! CHECK-NEXT:          %[[OD_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:            (%[[OD_J_I64]], %[[OD_I_I64]])
! CHECK-NEXT:          hlfir.assign %[[OD_SUM]] to %[[OD_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:             }
! CHECK-NEXT: %{{.*}} = fir.convert
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             fir.store {{.*}} to %[[OD_J_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_ordered_depth(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do ordered(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[C_I:.*]]:2 = hlfir.declare {{.*}}test_collapseEi
! CHECK: %[[C_J:.*]]:2 = hlfir.declare {{.*}}test_collapseEj
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:      %[[C_I]]#0 -> %[[C_I_PRIV:[^, ]+]],
! CHECK-SAME:      {{.*}}Ej_private_i32
! CHECK-SAME:      %[[C_J]]#0 -> %[[C_J_PRIV:[^ ]+]] :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[C_I_IV:[^, ]+]], %[[C_J_IV:[^) ]+]])
! CHECK-SAME:        collapse(2)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[C_I_DECL:.*]]:2 = hlfir.declare %[[C_I_PRIV]]
! CHECK:             %[[C_J_DECL:.*]]:2 = hlfir.declare %[[C_J_PRIV]]
! CHECK:             hlfir.assign %[[C_I_IV]] to %[[C_I_DECL]]#0
! CHECK:             hlfir.assign %[[C_J_IV]] to %[[C_J_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[C_I_VALUE:.*]] = fir.load %[[C_I_DECL]]#0
! CHECK:             %[[C_J_VALUE:.*]] = fir.load %[[C_J_DECL]]#0
! CHECK:             %[[C_SUM:.*]] = arith.addi
! CHECK-SAME:          %[[C_I_VALUE]], %[[C_J_VALUE]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[C_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign %[[C_SUM]] to %[[C_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_collapse(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do collapse(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! SIMD collapse makes every affected index lastprivate in OpenMP 5.2. Check
! that lowering copies both private values back to their source bindings.
! CHECK-LABEL: func.func @_QPtest_simd_collapse_lastprivate(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[SC_N:.*]]:2 = hlfir.declare {{.*}}lastprivateEn
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[SC_I:.*]]:2 = hlfir.declare {{.*}}lastprivateEi
! CHECK: %[[SC_J:.*]]:2 = hlfir.declare {{.*}}lastprivateEj
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.simd private({{[^,]*}}Ei_private_i32
! CHECK-SAME:      %[[SC_I]]#0 -> %[[SC_I_PRIV:[^, ]+]],
! CHECK-SAME:      {{.*}}Ej_private_i32
! CHECK-SAME:      %[[SC_J]]#0 -> %[[SC_J_PRIV:[^ ]+]] :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[SC_I_IV:[^, ]+]], %[[SC_J_IV:[^) ]+]])
! CHECK-SAME:        collapse(2)
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[SC_I_DECL:.*]]:2 = hlfir.declare %[[SC_I_PRIV]]
! CHECK:             %[[SC_J_DECL:.*]]:2 = hlfir.declare %[[SC_J_PRIV]]
! CHECK:             hlfir.assign %[[SC_I_IV]] to %[[SC_I_DECL]]#0
! CHECK:             hlfir.assign %[[SC_J_IV]] to %[[SC_J_DECL]]#0
! CHECK-NEXT:        %[[SC_I_VALUE:.*]] = fir.load %[[SC_I_DECL]]#0
! CHECK-NEXT:        %[[SC_J_VALUE:.*]] = fir.load %[[SC_J_DECL]]#0
! CHECK-NEXT:        %[[SC_SUM:.*]] = arith.addi
! CHECK-SAME:          %[[SC_I_VALUE]], %[[SC_J_VALUE]]
! CHECK-NEXT:        %[[SC_J_INDEX:.*]] = fir.load %[[SC_J_DECL]]#0
! CHECK-NEXT:        %[[SC_J_I64:.*]] = fir.convert %[[SC_J_INDEX]]
! CHECK-NEXT:        %[[SC_I_INDEX:.*]] = fir.load %[[SC_I_DECL]]#0
! CHECK-NEXT:        %[[SC_I_I64:.*]] = fir.convert %[[SC_I_INDEX]]
! CHECK-NEXT:        %[[SC_ELEMENT:.*]] = hlfir.designate
! CHECK-SAME:          (%[[SC_J_I64]], %[[SC_I_I64]])
! CHECK-NEXT:        hlfir.assign %[[SC_SUM]] to %[[SC_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_I_BOUND:.*]] = fir.load %[[SC_N]]#0
! CHECK-NEXT:        %[[SC_I_STEP:.*]] = arith.constant 1 : i32
! CHECK-NEXT:        %[[SC_J_BOUND:.*]] = fir.load %[[SC_N]]#0
! CHECK-NEXT:        %[[SC_J_STEP:.*]] = arith.constant 1 : i32
! CHECK-NEXT:        %[[SC_I_NEXT:.*]] = arith.addi
! CHECK-SAME:          %[[SC_I_IV]], %[[SC_I_STEP]]
! CHECK-NEXT:        %[[SC_I_ZERO:.*]] = arith.constant 0 : i32
! CHECK-NEXT:        %[[SC_I_NEG:.*]] = arith.cmpi slt,
! CHECK-SAME:          %[[SC_I_STEP]], %[[SC_I_ZERO]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_I_LT:.*]] = arith.cmpi slt,
! CHECK-SAME:          %[[SC_I_NEXT]], %[[SC_I_BOUND]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_I_GT:.*]] = arith.cmpi sgt,
! CHECK-SAME:          %[[SC_I_NEXT]], %[[SC_I_BOUND]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_I_LAST:.*]] = arith.select
! CHECK-SAME:          %[[SC_I_NEG]], %[[SC_I_LT]], %[[SC_I_GT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_J_NEXT:.*]] = arith.addi
! CHECK-SAME:          %[[SC_J_IV]], %[[SC_J_STEP]]
! CHECK-NEXT:        %[[SC_J_ZERO:.*]] = arith.constant 0 : i32
! CHECK-NEXT:        %[[SC_J_NEG:.*]] = arith.cmpi slt,
! CHECK-SAME:          %[[SC_J_STEP]], %[[SC_J_ZERO]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_J_LT:.*]] = arith.cmpi slt,
! CHECK-SAME:          %[[SC_J_NEXT]], %[[SC_J_BOUND]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_J_GT:.*]] = arith.cmpi sgt,
! CHECK-SAME:          %[[SC_J_NEXT]], %[[SC_J_BOUND]] : i32
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_J_LAST:.*]] = arith.select
! CHECK-SAME:          %[[SC_J_NEG]], %[[SC_J_LT]], %[[SC_J_GT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[SC_LAST:.*]] = arith.andi
! CHECK-SAME:          %[[SC_I_LAST]], %[[SC_J_LAST]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             fir.if %[[SC_LAST]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[SC_I_NEXT]] to %[[SC_I_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[SC_J_NEXT]] to %[[SC_J_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[SC_I_COPY:.*]] = fir.load %[[SC_I_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[SC_I_COPY]] to %[[SC_I]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[SC_J_COPY:.*]] = fir.load %[[SC_J_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               hlfir.assign %[[SC_J_COPY]] to %[[SC_J]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:             }
! CHECK-NEXT: omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_simd_collapse_lastprivate(n, a)
  integer :: n, a(n, n), i, j
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: simd collapse(2)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    do j = 1, n
      a(j, i) = i + j
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_block_nested_do(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[BLOCK_I:.*]]:2 = hlfir.declare {{.*}}block_nested_doEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:      %[[BLOCK_I]]#0 -> %[[BLOCK_PRIV:.*]] :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[BLOCK_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[BLOCK_DECL:.*]]:2 = hlfir.declare %[[BLOCK_PRIV]]
! CHECK:             hlfir.assign %[[BLOCK_IV]] to %[[BLOCK_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[BLOCK_VALUE:.*]] = fir.load %[[BLOCK_DECL]]#0
! CHECK:             %[[BLOCK_INDEX:.*]] = fir.load %[[BLOCK_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             %[[BLOCK_ELEMENT:.*]] = hlfir.designate
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             hlfir.assign %[[BLOCK_VALUE]] to %[[BLOCK_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_block_nested_do(n, a)
  integer :: n, a(n), i
  block
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: do) &
    !$omp & otherwise(nothing)
    do i = 1, n
      a(i) = i
    end do
  end block
end subroutine

! The selected-loop IV exception must not claim an IV predetermined by an
! enclosing construct when lowering an existing block-associated replacement.
! CHECK-LABEL: func.func @_QPtest_enclosing_do_iv(
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK: %[[ENC_I:.*]]:2 = hlfir.declare {{.*}}enclosing_do_ivEi
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         omp.wsloop private({{[^,]*}}Ei_private_i32
! CHECK-SAME:      %[[ENC_I]]#0 -> %[[ENC_PRIV:.*]] :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:           omp.loop_nest (%[[ENC_IV:.*]]) :
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             %[[ENC_DECL:.*]]:2 = hlfir.declare %[[ENC_PRIV]]
! CHECK:             hlfir.assign %[[ENC_IV]] to %[[ENC_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:         {{^ *}}}
! CHECK:             omp.parallel {
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[ENC_VALUE:.*]] = fir.load %[[ENC_DECL]]#0
! CHECK:               %[[ENC_INDEX:.*]] = fir.load %[[ENC_DECL]]#0
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               %[[ENC_ELEMENT:.*]] = hlfir.designate
! CHECK:               hlfir.assign %[[ENC_VALUE]] to %[[ENC_ELEMENT]]
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK-NOT:           {{^ *}}}
! CHECK:               omp.terminator
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:             omp.yield
! CHECK-NOT:     {{omp\.(wsloop|simd|loop_nest)|fir\.do_loop}}
! CHECK:         return
subroutine test_enclosing_do_iv(n, a)
  integer :: n, a(n), i
  !$omp do
  do i = 1, n
    !$omp begin metadirective &
    !$omp& when(implementation={vendor(llvm)}: parallel) &
    !$omp& otherwise(nothing)
    a(i) = i
    !$omp end metadirective
  end do
end subroutine

! A predetermined flag left by a sibling transform must likewise remain
! outside an existing non-loop replacement.
! CHECK-LABEL: func.func @_QPtest_sibling_transform_single(
! CHECK: %[[SIB_I:.*]]:2 = hlfir.declare {{.*}}transform_singleEi
! CHECK:         omp.canonical_loop
! CHECK:         omp.unroll_partial
! CHECK:         omp.single {
! CHECK-NOT:       {{omp\.(canonical_loop|wsloop|simd|loop_nest)}}
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:           %[[SIB_VALUE:.*]] = fir.load %[[SIB_I]]#0
! CHECK-NOT:       {{omp\.(canonical_loop|wsloop|simd|loop_nest)}}
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:           %[[SIB_ELEMENT:.*]] = hlfir.designate
! CHECK:           hlfir.assign %[[SIB_VALUE]] to %[[SIB_ELEMENT]]
! CHECK-NOT:       {{omp\.(canonical_loop|wsloop|simd|loop_nest)}}
! CHECK-NOT:       fir.do_loop
! CHECK-NOT:       {{^ *}}}
! CHECK:           omp.terminator
! CHECK-NOT:     {{omp\.(canonical_loop|wsloop|simd|loop_nest)}}
! CHECK-NOT:     fir.do_loop
! CHECK:         return
subroutine test_sibling_transform_single(n, a)
  integer :: n, a(n), i
  !$omp unroll partial(2)
  do i = 1, n
    a(i) = i
  end do
  !$omp begin metadirective &
  !$omp& when(user={condition(.true.)}: single) &
  !$omp& otherwise(nothing)
  a(1) = i
  !$omp end metadirective
end subroutine
