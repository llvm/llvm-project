! This test checks lowering of OpenACC declare directive in function and
! subroutine specification parts.

! RUN: bbc -fopenacc -emit-hlfir --openacc-unwrap-fir-box=true --openacc-generate-default-bounds=true %s -o - | FileCheck %s

module acc_declare
  contains

  subroutine acc_declare_copy()
    integer :: a(100), i
    !$acc declare copy(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_copy()
! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyEa"}
! CHECK-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copy>, uniq_name = "_QMacc_declareFacc_declare_copyEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[BOUND:.*]] = acc.bounds   lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {
! CHECK: }
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) to varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK: return

  subroutine acc_declare_create()
    integer :: a(100), i
    !$acc declare create(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_create() {
! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_createEa"}
! CHECK-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_create>, uniq_name = "_QMacc_declareFacc_declare_createEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[BOUND:.*]] = acc.bounds   lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {
! CHECK: }
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: acc.delete accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) {dataClause = #acc<data_clause acc_create>, name = "a"}
! CHECK: return

  subroutine acc_declare_present(a)
    integer :: a(100), i
    !$acc declare present(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_present(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
! CHECK-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_present>, uniq_name = "_QMacc_declareFacc_declare_presentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%[[C1]] : index)
! CHECK: %[[PRESENT:.*]] = acc.present varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: acc.declare_enter dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_copyin()
    integer :: a(100), b(10), i
    !$acc declare copyin(a) copyin(readonly: b)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_copyin()
! CHECK: %[[A:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyinEa"}
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin>, uniq_name = "_QMacc_declareFacc_declare_copyinEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[B:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "b", uniq_name = "_QMacc_declareFacc_declare_copyinEb"}
! CHECK: %[[BDECL:.*]]:2 = hlfir.declare %[[B]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin_readonly>, uniq_name = "_QMacc_declareFacc_declare_copyinEb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK: %[[BOUND_A:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ADECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND_A]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: %[[BOUND_B:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[BDECL]]#0 : !fir.ref<!fir.array<10xi32>>)   bounds(%[[BOUND_B]]) -> !fir.ref<!fir.array<10xi32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! CHECK: acc.declare_enter dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<10xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)
! CHECK: acc.delete accPtr(%[[COPYIN_A]] : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND_A]]) {dataClause = #acc<data_clause acc_copyin>, name = "a"}
! CHECK: acc.delete accPtr(%[[COPYIN_B]] : !fir.ref<!fir.array<10xi32>>)   bounds(%[[BOUND_B]]) {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}

  subroutine acc_declare_copyout()
    integer :: a(100), i
    !$acc declare copyout(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_copyout()
! CHECK: %[[A:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyoutEa"}
! CHECK: %[[ADECL:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyout>, uniq_name = "_QMacc_declareFacc_declare_copyoutEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[ADECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: acc.copyout accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) to varPtr(%[[ADECL]]#0 : !fir.ref<!fir.array<100xi32>>) {name = "a"}
! CHECK: return

  subroutine acc_declare_deviceptr(a)
    integer :: a(100), i
    !$acc declare deviceptr(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_deviceptr(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}) {
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptrEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: acc.declare_enter dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_link(a)
    integer :: a(100), i
    !$acc declare link(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_link(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_linkEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: acc.declare_enter dataOperands(%[[LINK]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_device_resident(a)
    integer :: a(100), i
    !$acc declare device_resident(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_device_resident(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_residentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xi32>>)
! CHECK: acc.delete accPtr(%[[DEVICERES]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_declare_device_resident>, name = "a"}

  subroutine acc_declare_device_resident2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare device_resident(dataparam)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_device_resident2()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_device_resident2Edataparam"}
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_resident2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xf32>>)
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xf32>>)
! CHECK: acc.delete accPtr(%[[DEVICERES]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_declare_device_resident>, name = "dataparam"}

  subroutine acc_declare_link2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare link(dataparam)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_link2()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_link2Edataparam"}
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_link2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! CHECK: acc.declare_enter dataOperands(%[[LINK]] : !fir.ref<!fir.array<100xf32>>)

  subroutine acc_declare_deviceptr2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare deviceptr(dataparam)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_deviceptr2()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_deviceptr2Edataparam"}
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptr2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]]#0 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! CHECK: acc.declare_enter dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100xf32>>)

  function acc_declare_in_func()
    real :: a(1024)
    !$acc declare device_resident(a)
  end function

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_in_func() -> f32 {
! CHECK: %[[DEVICE_RESIDENT:.*]] = acc.declare_device_resident varPtr(%{{.*}}#0 : !fir.ref<!fir.array<1024xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<1024xf32>> {name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[DEVICE_RESIDENT]] : !fir.ref<!fir.array<1024xf32>>)
! CHECK: %[[LOAD:.*]] = fir.load %{{.*}}#1 : !fir.ref<f32>
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[DEVICE_RESIDENT]] : !fir.ref<!fir.array<1024xf32>>)
! CHECK: acc.delete accPtr(%[[DEVICE_RESIDENT]] : !fir.ref<!fir.array<1024xf32>>) bounds(%6) {dataClause = #acc<data_clause acc_declare_device_resident>, name = "a"}
! CHECK: return %[[LOAD]] : f32
! CHECK: }

  function acc_declare_in_func2(i)
    real :: a(1024)
    integer :: i
    !$acc declare create(a)
    return
  end function

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_in_func2(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) -> f32 {
! CHECK: %[[ALLOCA_A:.*]] = fir.alloca !fir.array<1024xf32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_in_func2Ea"}
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ALLOCA_A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_create>, uniq_name = "_QMacc_declareFacc_declare_in_func2Ea"} : (!fir.ref<!fir.array<1024xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1024xf32>>, !fir.ref<!fir.array<1024xf32>>)
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[DECL_A]]#0 : !fir.ref<!fir.array<1024xf32>>) bounds(%{{[0-9]+}}) -> !fir.ref<!fir.array<1024xf32>> {name = "a"}
! CHECK: %[[TOKEN:.*]] = acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<1024xf32>>)
! CHECK:   cf.br ^bb1
! CHECK: ^bb1:
! CHECK: acc.declare_exit token(%[[TOKEN]]) dataOperands(%[[CREATE]] : !fir.ref<!fir.array<1024xf32>>)
! CHECK: acc.delete accPtr(%[[CREATE]] : !fir.ref<!fir.array<1024xf32>>) bounds(%{{[0-9]+}}) {dataClause = #acc<data_clause acc_create>, name = "a"}
! CHECK:   return %{{.*}} : f32
! CHECK: }

  subroutine acc_declare_allocate()
    integer, allocatable :: a(:)
    !$acc declare create(a)

    allocate(a(100))

! CHECK: %{{.*}} = fir.allocmem !fir.array<?xi32>, %{{.*}} {fir.must_be_heap = true, uniq_name = "_QMacc_declareFacc_declare_allocateEa.alloc"}
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_alloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

    deallocate(a)

! CHECK: %{{.*}} = fir.box_addr %{{.*}} {acc.declare_action = #acc.declare_action<preDealloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_pre_dealloc>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>

! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.array<?xi32>>
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postDealloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_dealloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! CHECK: fir.if
! CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.array<?xi32>>
! CHECK: fir.store %{{.*}} to %{{.*}}#1 {acc.declare_action = #acc.declare_action<postDealloc = @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_dealloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: }

  end subroutine

! CHECK-LABEL: func.func private @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_alloc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:         %[[UPDATE:.*]] = acc.update_device varPtr(%[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {implicit = true, name = "a_desc", structured = false}
! CHECK:         acc.update dataOperands(%[[UPDATE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] {acc.declare = #acc.declare<dataClause =  acc_create>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> {name = "a", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func.func private @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_pre_dealloc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:         %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] {acc.declare = #acc.declare<dataClause =  acc_create>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[GETDEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> {dataClause = #acc<data_clause acc_create>, name = "a", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[GETDEVICEPTR]] : !fir.heap<!fir.array<?xi32>>)
! CHECK:         acc.delete accPtr(%[[GETDEVICEPTR]] : !fir.heap<!fir.array<?xi32>>) {dataClause = #acc<data_clause acc_create>, name = "a", structured = false}
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func.func private @_QMacc_declareFacc_declare_allocateEa_acc_declare_update_desc_post_dealloc(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! CHECK:         %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[UPDATE:.*]] = acc.update_device varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> {implicit = true, name = "a_desc", structured = false}
! CHECK:         acc.update dataOperands(%[[UPDATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK:         return
! CHECK:       }

  subroutine acc_declare_multiple_directive(a, b)
    integer :: a(100), b(100), i
    !$acc declare copy(a)
    !$acc declare copyout(b)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_multiple_directive(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "b"}) {
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_copy>, uniq_name = "_QMacc_declareFacc_declare_multiple_directiveEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[DECL_B:.*]]:2 = hlfir.declare %[[ARG1]](%{{.*}}) dummy_scope %{{[0-9]+}} {acc.declare = #acc.declare<dataClause =  acc_copyout>, uniq_name = "_QMacc_declareFacc_declare_multiple_directiveEb"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL_A]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! CHECK: %[[CREATE:.*]] = acc.create varPtr(%[[DECL_B]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copyout>, name = "b"}
! CHECK: acc.declare_enter dataOperands(%[[COPYIN]], %[[CREATE]] : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK: %{{.*}}:{{.*}} = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {


! CHECK: acc.copyout accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) to varPtr(%[[DECL_B]]#0 : !fir.ref<!fir.array<100xi32>>) {name = "b"}
! CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) to varPtr(%[[DECL_A]]#0 : !fir.ref<!fir.array<100xi32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}

  subroutine acc_declare_array_section(a)
    integer :: a(:)
    !$acc declare copy(a(1:10))

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_array_section(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QMacc_declareFacc_declare_array_sectionEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL_A]]#0 {acc.declare = #acc.declare<dataClause =  acc_copy>} : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[COPYIN:.*]] = acc.copyin varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {dataClause = #acc<data_clause acc_copy>, name = "a(1:10)"}
! CHECK: acc.declare_enter dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<?xi32>>)

! CHECK: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) to varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xi32>>) {dataClause = #acc<data_clause acc_copy>, name = "a(1:10)"}

  subroutine acc_declare_allocate_with_stat()
    integer :: status
    real, pointer, dimension(:) :: localptr
    !$acc declare create(localptr)
    allocate(localptr(n), stat=status)

    deallocate(localptr, stat=status)
  end subroutine

! CHECK-LABEL: func.func @_QMacc_declarePacc_declare_allocate_with_stat()
! CHECK: fir.call @_FortranAPointerAllocate(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declareFacc_declare_allocate_with_statElocalptr_acc_declare_update_desc_post_alloc>}
! CHECK: fir.call @_FortranAPointerDeallocate(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} {acc.declare_action = #acc.declare_action<preDealloc = @_QMacc_declareFacc_declare_allocate_with_statElocalptr_acc_declare_update_desc_pre_dealloc, postDealloc = @_QMacc_declareFacc_declare_allocate_with_statElocalptr_acc_declare_update_desc_post_dealloc>}
end module

module acc_declare_allocatable_test
 integer, allocatable :: data1(:)
 !$acc declare create(data1)
end module

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_allocatable_testEdata1_acc_ctor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[COPYIN:.*]] = acc.copyin varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {dataClause = #acc<data_clause acc_create>, implicit = true, name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[COPYIN]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         acc.terminator
! CHECK:       }

! CHECK-LABEL: func.func private @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_post_alloc() {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[UPDATE:.*]] = acc.update_device varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {implicit = true, name = "data1_desc", structured = false}
! CHECK:         acc.update dataOperands(%[[UPDATE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         %[[LOAD:.*]] = fir.load %[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[BOXADDR:.*]] = fir.box_addr %[[LOAD]] {acc.declare = #acc.declare<dataClause =  acc_create>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[BOXADDR]] : !fir.heap<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> {name = "data1", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func.func private @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_pre_dealloc() {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[LOAD:.*]] = fir.load %[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[BOXADDR:.*]] = fir.box_addr %[[LOAD]] {acc.declare = #acc.declare<dataClause =  acc_create>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         %[[DEVPTR:.*]] = acc.getdeviceptr varPtr(%[[BOXADDR]] : !fir.heap<!fir.array<?xi32>>)   -> !fir.heap<!fir.array<?xi32>> {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVPTR]] : !fir.heap<!fir.array<?xi32>>)
! CHECK:         acc.delete accPtr(%[[DEVPTR]] : !fir.heap<!fir.array<?xi32>>) {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func.func private @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_post_dealloc() {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[UPDATE:.*]] = acc.update_device varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {implicit = true, name = "data1_desc", structured = false}
! CHECK:         acc.update dataOperands(%[[UPDATE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: acc.global_dtor @_QMacc_declare_allocatable_testEdata1_acc_dtor {
! CHECK:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! CHECK:         acc.terminator
! CHECK:       }


module acc_declare_equivalent
  integer, parameter :: n = 10
  real :: v1(n)
  real :: v2(n)
  equivalence(v1(1), v2(1))
  !$acc declare create(v2)
end module

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_equivalentEv2_acc_ctor {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMacc_declare_equivalentEv1) {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.ref<!fir.array<40xi8>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[ADDR]] : !fir.ref<!fir.array<40xi8>>) -> !fir.ref<!fir.array<40xi8>> {name = "v2", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<40xi8>>)
! CHECK:         acc.terminator
! CHECK:       }
! CHECK-LABEL: acc.global_dtor @_QMacc_declare_equivalentEv2_acc_dtor {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMacc_declare_equivalentEv1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<40xi8>>
! CHECK:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[ADDR]] : !fir.ref<!fir.array<40xi8>>) -> !fir.ref<!fir.array<40xi8>> {dataClause = #acc<data_clause acc_create>, name = "v2", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<40xi8>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.array<40xi8>>) {dataClause = #acc<data_clause acc_create>, name = "v2", structured = false}
! CHECK:         acc.terminator
! CHECK:       }

module acc_declare_equivalent2
  real :: v1(10)
  real :: v2(5)
  equivalence(v1(6), v2(1))
  !$acc declare create(v2)
end module

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_equivalent2Ev2_acc_ctor {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMacc_declare_equivalent2Ev1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<40xi8>>
! CHECK:         %[[CREATE:.*]] = acc.create varPtr(%[[ADDR]] : !fir.ref<!fir.array<40xi8>>) -> !fir.ref<!fir.array<40xi8>> {name = "v2", structured = false}
! CHECK:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.ref<!fir.array<40xi8>>)
! CHECK:         acc.terminator
! CHECK:       }
! CHECK-LABEL: acc.global_dtor @_QMacc_declare_equivalent2Ev2_acc_dtor {
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QMacc_declare_equivalent2Ev1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.array<40xi8>>
! CHECK:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[ADDR]] : !fir.ref<!fir.array<40xi8>>) -> !fir.ref<!fir.array<40xi8>> {dataClause = #acc<data_clause acc_create>, name = "v2", structured = false}
! CHECK:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<40xi8>>)
! CHECK:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.array<40xi8>>) {dataClause = #acc<data_clause acc_create>, name = "v2", structured = false}
! CHECK:         acc.terminator
! CHECK:       }

! Test that the pre/post alloc/dealloc attributes are set when the
! allocate/deallocate statement are in a different module.
module acc_declare_allocatable_test2
contains
  subroutine init()
    use acc_declare_allocatable_test
    allocate(data1(100))
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_post_alloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  end subroutine

  subroutine finalize()
    use acc_declare_allocatable_test
    deallocate(data1)
! CHECK: %{{.*}} = fir.box_addr %{{.*}} {acc.declare_action = #acc.declare_action<preDealloc = @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_pre_dealloc>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK: fir.store %{{.*}} to %{{.*}} {acc.declare_action = #acc.declare_action<postDealloc = @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_post_dealloc>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  end subroutine
end module

module acc_declare_allocatable_test3
  integer, allocatable :: data1(:)
  integer, allocatable :: data2(:)
  !$acc declare create(data1, data2, data1)
end module

! CHECK-LABEL: acc.global_ctor @_QMacc_declare_allocatable_test3Edata1_acc_ctor
! CHECK-LABEL: acc.global_ctor @_QMacc_declare_allocatable_test3Edata2_acc_ctor

module acc_declare_post_action_stat
  real, dimension(:), allocatable :: x, y
  !$acc declare create(x,y)

contains

  subroutine init()
    integer :: stat
    allocate(x(10), y(10), stat=stat)
  end subroutine
end module

! CHECK-LABEL: func.func @_QMacc_declare_post_action_statPinit()
! CHECK: fir.call @_FortranAAllocatableAllocate({{.*}}) fastmath<contract> {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_post_action_statEx_acc_declare_update_desc_post_alloc>} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK: fir.if
! CHECK: fir.call @_FortranAAllocatableAllocate({{.*}}) fastmath<contract> {acc.declare_action = #acc.declare_action<postAlloc = @_QMacc_declare_post_action_statEy_acc_declare_update_desc_post_alloc>} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
