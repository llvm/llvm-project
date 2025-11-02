! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present for strings

! RUN: bbc -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 \
! RUN: | FileCheck %s

!CHECK:  omp.private {type = private} @{{.*}}test_allocatable_fixed_len_stringEfixed_len_str{{.*}} init {
!CHECK:    fir.if {{.*}} {
!CHECK:      fir.embox %{{[^[:space:]]+}} : {{.*}}
!CHECK:    } else {
!CHECK:      fir.embox %{{[^[:space:]]+}} : {{.*}}
!CHECK:    }
!CHECK:  }

!CHECK:  omp.private {type = private} @[[STR_ARR_PRIVATIZER:_QFtest_allocatable_string_arrayEc_private_box_heap_Uxc8xU]] : [[TYPE:.*]] init {
!CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<[[TYPE]]>, %[[C_PVT_BOX_REF:.*]]: !fir.ref<[[TYPE]]>):
!CHECK:      %{{.*}} = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:      } else {
!CHECK:        %[[C_PVT_ALLOC:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%{{.*}} : index), %{{.*}}
!CHECK:        %[[C_PVT_BOX:.*]] = fir.rebox
!CHECK:        fir.store %[[C_PVT_BOX]] to %[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      }
!CHECK:      omp.yield(%[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
!CHECK:  } dealloc {
!CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>):
!CHECK:      %{{.*}} = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:        fir.freemem %{{.*}} : !fir.heap<!fir.array<?x!fir.char<1,?>>>
!CHECK:      }
!CHECK:      omp.yield
!CHECK:  }

!CHECK:  omp.private {type = private} @[[STR_PRIVATIZER:_QFtest_allocatable_stringEc_private_box_heap_c8xU]] : [[TYPE:!fir.box<!fir.heap<!fir.char<1,\?>>>]] init {
!CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<[[TYPE]]>, %[[C_PVT_BOX_REF:.*]]: !fir.ref<[[TYPE]]>):
!CHECK:      %[[C_BOX:.*]] = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:        %[[C_PVT_MEM:.*]] = fir.allocmem !fir.char<1,?>(%{{.*}} : index)
!CHECK:        %[[C_PVT_BOX:.*]] = fir.embox %[[C_PVT_MEM]] typeparams %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
!CHECK:        fir.store %[[C_PVT_BOX]] to %[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:      }
!CHECK:      omp.yield(%[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:  } dealloc {
!CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>):
!CHECK:      fir.if %{{.*}} {
!CHECK:        fir.freemem %{{.*}} : !fir.heap<!fir.char<1,?>>
!CHECK:      }
!CHECK:      omp.yield
!CHECK:  }

!CHECK:  func.func @_QPtest_allocatable_string(%{{.*}}: !fir.ref<i32> {fir.bindc_name = "n"}) {
!CHECK:    %[[C_BOX_REF:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "c", uniq_name = "_QFtest_allocatable_stringEc"}
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_BOX_REF]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_stringEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, i32) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:    omp.parallel private(@[[STR_PRIVATIZER]] %[[C_DECL]]#0 -> %[[PRIV_ARG:.*]] : {{.*}}) {
!CHECK:      hlfir.declare %[[PRIV_ARG]]
!CHECK:      omp.terminator
!CHECK:    }
subroutine test_allocatable_string(n)
  character(n), allocatable :: c
  !$omp parallel private(c)
  !$omp end parallel
end subroutine

!CHECK:  func.func @_QPtest_allocatable_string_array(%[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
!CHECK:    %{{.*}} = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest_allocatable_string_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[C_BOX_REF:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {bindc_name = "c", uniq_name = "_QFtest_allocatable_string_arrayEc"}
!CHECK:    %[[C_BOX:.*]] = fir.embox %{{.*}}(%{{.*}}) typeparams %{{.*}} : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
!CHECK:    fir.store %[[C_BOX]] to %[[C_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_BOX_REF]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_string_arrayEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, i32) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
!CHECK:    omp.parallel private(@[[STR_ARR_PRIVATIZER]] %[[C_DECL]]#0 -> %[[PRIV_ARG:.*]] : {{.*}}) {
!CHECK:      hlfir.declare %[[PRIV_ARG]]
!CHECK:      omp.terminator
!CHECK:    }

subroutine test_allocatable_string_array(n)
  character(n), allocatable :: c(:)
  !$omp parallel private(c)
  !$omp end parallel
end subroutine

subroutine test_allocatable_fixed_len_string()
  character(42), allocatable :: fixed_len_str
  !$omp parallel do private(fixed_len_str)
  do i = 1,10
  end do
  !$omp end parallel do
end subroutine
