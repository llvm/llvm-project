! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present for strings

! REQUIRES: shell
! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK:  func.func @_QPtest_allocatable_string(%{{.*}}: !fir.ref<i32> {fir.bindc_name = "n"}) {
!CHECK:    %[[C_BOX_REF:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "c", uniq_name = "_QFtest_allocatable_stringEc"}
!CHECK:    %[[C_DECL:.*]]:2 = hlfir.declare %[[C_BOX_REF]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_stringEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, i32) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:    omp.parallel {
!CHECK:      %[[C_PVT_BOX_REF:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "c", pinned, uniq_name = "_QFtest_allocatable_stringEc"}
!CHECK:      %[[C_BOX:.*]] = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:        %[[C_PVT_MEM:.*]] = fir.allocmem !fir.char<1,?>(%{{.*}} : index) {fir.must_be_heap = true, uniq_name = "_QFtest_allocatable_stringEc.alloc"}
!CHECK:        %[[C_PVT_BOX:.*]] = fir.embox %[[C_PVT_MEM]] typeparams %{{.*}} : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
!CHECK:        fir.store %[[C_PVT_BOX]] to %[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:      }
!CHECK:      %[[C_PVT_DECL:.*]]:2 = hlfir.declare %[[C_PVT_BOX_REF]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_stringEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:      fir.if %{{.*}} {
!CHECK:        %[[C_PVT_BOX:.*]] = fir.load %[[C_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:        %[[C_PVT_BOX_ADDR:.*]] = fir.box_addr %[[C_PVT_BOX]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
!CHECK:        fir.freemem %[[C_PVT_BOX_ADDR]] : !fir.heap<!fir.char<1,?>>
!CHECK:      }
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
!CHECK:    omp.parallel {
!CHECK:      %[[C_PVT_BOX_REF:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>> {bindc_name = "c", pinned, uniq_name = "_QFtest_allocatable_string_arrayEc"}
!CHECK:      %{{.*}} = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:        %[[C_PVT_ALLOC:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%{{.*}} : index), %{{.*}} {fir.must_be_heap = true, uniq_name = "_QFtest_allocatable_string_arrayEc.alloc"}
!CHECK:        %[[C_PVT_BOX:.*]] = fir.embox %[[C_PVT_ALLOC]](%{{.*}}) typeparams %{{.*}} : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shapeshift<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
!CHECK:        fir.store %[[C_PVT_BOX]] to %[[C_PVT_BOX_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      }
!CHECK:      %[[C_PVT_DECL:.*]]:2 = hlfir.declare %[[C_PVT_BOX_REF]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_string_arrayEc"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>)
!CHECK:      %{{.*}} = fir.load %[[C_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:      fir.if %{{.*}} {
!CHECK:        %[[C_PVT_BOX:.*]] = fir.load %[[C_PVT_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
!CHECK:        %[[C_PVT_ADDR:.*]] = fir.box_addr %[[C_PVT_BOX]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
!CHECK:        fir.freemem %[[C_PVT_ADDR]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
!CHECK:      }
!CHECK:      omp.terminator
!CHECK:    }

subroutine test_allocatable_string_array(n)
  character(n), allocatable :: c(:)
  !$omp parallel private(c)
  !$omp end parallel
end subroutine
