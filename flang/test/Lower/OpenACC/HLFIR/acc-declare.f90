! This test checks lowering of OpenACC declare directive in function and
! subroutine specification parts.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=HLFIR,ALL
! RUN: bbc -fopenacc -emit-fir -hlfir %s -o - | FileCheck %s --check-prefixes=FIR,ALL

module acc_declare
  contains

  subroutine acc_declare_copy()
    integer :: a(100), i
    !$acc declare copy(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_copy()
! ALL-DAG: %[[C1:.*]] = arith.constant 1 : index
! ALL-DAG: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyEa"}
! FIR-DAG: %[[DECL:.*]] = fir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copy>, uniq_name = "_QMacc_declareFacc_declare_copyEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copy>, uniq_name = "_QMacc_declareFacc_declare_copyEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! ALL: %[[BOUND:.*]] = acc.bounds   lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! FIR: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: %[[COPYIN:.*]] = acc.copyin varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copy>, name = "a"}
! ALL: acc.declare dataOperands(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>)

! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {
! ALL: }

! FIR: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) to varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}
! HLFIR: acc.copyout accPtr(%[[COPYIN]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) to varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>) {dataClause = #acc<data_clause acc_copy>, name = "a"}

! ALL: return

  subroutine acc_declare_create()
    integer :: a(100), i
    !$acc declare create(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_create() {
! ALL-DAG: %[[C1:.*]] = arith.constant 1 : index
! ALL-DAG: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_createEa"}
! FIR-DAG: %[[DECL:.*]] = fir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_create>, uniq_name = "_QMacc_declareFacc_declare_createEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_create>, uniq_name = "_QMacc_declareFacc_declare_createEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! ALL: %[[BOUND:.*]] = acc.bounds   lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! FIR: %[[CREATE:.*]] = acc.create varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[CREATE:.*]] = acc.create varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: acc.declare dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)

! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index, i32) {
! ALL: }

! ALL: acc.delete accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) {dataClause = #acc<data_clause acc_create>, name = "a"}
! ALL: return

  subroutine acc_declare_present(a)
    integer :: a(100), i
    !$acc declare present(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_present(
! ALL-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! ALL-DAG: %[[C1:.*]] = arith.constant 1 : index
! FIR-DAG: %[[DECL:.*]] = fir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_present>, uniq_name = "_QMacc_declareFacc_declare_presentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR-DAG: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_present>, uniq_name = "_QMacc_declareFacc_declare_presentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! ALL: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%[[C1]] : index)
! FIR: %[[PRESENT:.*]] = acc.present varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[PRESENT:.*]] = acc.present varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: acc.declare dataOperands(%[[PRESENT]] : !fir.ref<!fir.array<100xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_copyin()
    integer :: a(100), b(10), i
    !$acc declare copyin(a) copyin(readonly: b)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_copyin()
! ALL: %[[A:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyinEa"}
! FIR: %[[ADECL:.*]] = fir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin>, uniq_name = "_QMacc_declareFacc_declare_copyinEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR: %[[ADECL:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin>, uniq_name = "_QMacc_declareFacc_declare_copyinEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! ALL: %[[B:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "b", uniq_name = "_QMacc_declareFacc_declare_copyinEb"}
! FIR: %[[BDECL:.*]] = fir.declare %[[B]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin_readonly>, uniq_name = "_QMacc_declareFacc_declare_copyinEb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
! HLFIR: %[[BDECL:.*]]:2 = hlfir.declare %[[B]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyin_readonly>, uniq_name = "_QMacc_declareFacc_declare_copyinEb"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
! ALL: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ADECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[COPYIN_A:.*]] = acc.copyin varPtr(%[[ADECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: %[[BOUND:.*]] = acc.bounds lowerbound(%{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! FIR: %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[BDECL]] : !fir.ref<!fir.array<10xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! HLFIR: %[[COPYIN_B:.*]] = acc.copyin varPtr(%[[BDECL]]#1 : !fir.ref<!fir.array<10xi32>>)   bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xi32>> {dataClause = #acc<data_clause acc_copyin_readonly>, name = "b"}
! ALL: acc.declare dataOperands(%[[COPYIN_A]], %[[COPYIN_B]] : !fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<10xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_copyout()
    integer :: a(100), i
    !$acc declare copyout(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_copyout()
! ALL: %[[A:.*]] = fir.alloca !fir.array<100xi32> {bindc_name = "a", uniq_name = "_QMacc_declareFacc_declare_copyoutEa"}
! FIR: %[[ADECL:.*]] = fir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyout>, uniq_name = "_QMacc_declareFacc_declare_copyoutEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR: %[[ADECL:.*]]:2 = hlfir.declare %[[A]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_copyout>, uniq_name = "_QMacc_declareFacc_declare_copyoutEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! FIR: %[[CREATE:.*]] = acc.create varPtr(%[[ADECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! HLFIR: %[[CREATE:.*]] = acc.create varPtr(%[[ADECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {dataClause = #acc<data_clause acc_copyout>, name = "a"}
! ALL: acc.declare dataOperands(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

! FIR: acc.copyout accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) to varPtr(%[[ADECL]] : !fir.ref<!fir.array<100xi32>>) {name = "a"}
! HLFIR: acc.copyout accPtr(%[[CREATE]] : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) to varPtr(%[[ADECL]]#1 : !fir.ref<!fir.array<100xi32>>) {name = "a"}
! ALL: return

  subroutine acc_declare_deviceptr(a)
    integer :: a(100), i
    !$acc declare deviceptr(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_deviceptr(
! ALL-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}) {
! FIR: %[[DECL:.*]] = fir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptrEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptrEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! FIR: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: acc.declare dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_link(a)
    integer :: a(100), i
    !$acc declare link(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_link(
! ALL-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! FIR: %[[DECL:.*]] = fir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_linkEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_linkEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! FIR: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: acc.declare dataOperands(%[[LINK]] : !fir.ref<!fir.array<100xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

  subroutine acc_declare_device_resident(a)
    integer :: a(100), i
    !$acc declare device_resident(a)

    do i = 1, 100
      a(i) = i
    end do
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_device_resident(
! ALL-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! FIR: %[[DECL:.*]] = fir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_residentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_residentEa"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! FIR: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! HLFIR: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xi32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "a"}
! ALL: acc.declare dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xi32>>)
! ALL: %{{.*}}:2 = fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%arg{{.*}} = %{{.*}}) -> (index, i32)

! ALL: acc.delete accPtr(%[[DEVICERES]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_declare_device_resident>, name = "a"}

  subroutine acc_declare_device_resident2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare device_resident(dataparam)
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_device_resident2()
! ALL: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_device_resident2Edataparam"}
! FIR: %[[DECL:.*]] = fir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_resident2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_device_resident>, uniq_name = "_QMacc_declareFacc_declare_device_resident2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! FIR: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! HLFIR: %[[DEVICERES:.*]] = acc.declare_device_resident varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! ALL: acc.declare dataOperands(%[[DEVICERES]] : !fir.ref<!fir.array<100xf32>>)

! ALL: acc.delete accPtr(%[[DEVICERES]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) {dataClause = #acc<data_clause acc_declare_device_resident>, name = "dataparam"}

  subroutine acc_declare_link2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare link(dataparam)
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_link2()
! ALL: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_link2Edataparam"}
! FIR: %[[DECL:.*]] = fir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_link2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_declare_link>, uniq_name = "_QMacc_declareFacc_declare_link2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! FIR: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! HLFIR: %[[LINK:.*]] = acc.declare_link varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! ALL: acc.declare dataOperands(%[[LINK]] : !fir.ref<!fir.array<100xf32>>)

  subroutine acc_declare_deviceptr2()
    integer, parameter :: n = 100
    real, dimension(n) :: dataparam
    !$acc declare deviceptr(dataparam)
  end subroutine

! ALL-LABEL: func.func @_QMacc_declarePacc_declare_deviceptr2()
! ALL: %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "dataparam", uniq_name = "_QMacc_declareFacc_declare_deviceptr2Edataparam"}
! FIR: %[[DECL:.*]] = fir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptr2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<100xf32>>
! HLFIR: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]](%{{.*}}) {acc.declare = #acc.declare<dataClause =  acc_deviceptr>, uniq_name = "_QMacc_declareFacc_declare_deviceptr2Edataparam"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! FIR: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! HLFIR: %[[DEVICEPTR:.*]] = acc.deviceptr varPtr(%[[DECL]]#1 : !fir.ref<!fir.array<100xf32>>)   bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "dataparam"}
! ALL: acc.declare dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.array<100xf32>>)

end module

module acc_declare_allocatable_test
 integer, allocatable :: data1(:)
 !$acc declare create(data1)
end module

! ALL-LABEL: acc.global_ctor @_QMacc_declare_allocatable_testEdata1_acc_ctor {
! ALL:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! ALL:         %[[COPYIN:.*]] = acc.copyin varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {dataClause = #acc<data_clause acc_create>, implicit = true, name = "data1", structured = false}
! ALL:         acc.declare_enter dataOperands(%[[COPYIN]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! ALL:         acc.terminator
! ALL:       }

! ALL-LABEL: func.func private @_QMacc_declare_allocatable_testEdata1_acc_declare_update_desc_post_alloc() {
! ALL:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! ALL:         %[[LOAD:.*]] = fir.load %[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! ALL:         %[[BOXADDR:.*]] = fir.box_addr %[[LOAD]] {acc.declare = #acc.declare<dataClause =  acc_create>} : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! ALL:         %[[CREATE:.*]] = acc.create varPtr(%[[BOXADDR]] : !fir.heap<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>> {name = "data1", structured = false}
! ALL:         acc.declare_enter dataOperands(%[[CREATE]] : !fir.heap<!fir.array<?xi32>>)
! ALL:         %[[UPDATE:.*]] = acc.update_device varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {implicit = true, name = "data1_desc", structured = false}
! ALL:         acc.update dataOperands(%[[UPDATE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! ALL:         return
! ALL:       }

! ALL-LABEL: acc.global_dtor @_QMacc_declare_allocatable_testEdata1_acc_dtor {
! ALL:         %[[GLOBAL_ADDR:.*]] = fir.address_of(@_QMacc_declare_allocatable_testEdata1) {acc.declare = #acc.declare<dataClause =  acc_create>} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! ALL:         %[[DEVICEPTR:.*]] = acc.getdeviceptr varPtr(%[[GLOBAL_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! ALL:         acc.declare_exit dataOperands(%[[DEVICEPTR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! ALL:         acc.delete accPtr(%[[DEVICEPTR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {dataClause = #acc<data_clause acc_create>, name = "data1", structured = false}
! ALL:         acc.terminator
! ALL:       }
