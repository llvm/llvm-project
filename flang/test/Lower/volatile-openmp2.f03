! RUN: bbc -fopenmp --strict-fir-volatile-verifier %s -o - | FileCheck %s
type x
    integer::x1=1
end type x
class(x),allocatable,volatile,asynchronous::v(:)
!$omp parallel private(v)
!$omp end parallel
end

! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QFE.n.x1) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] typeparams %[[VAL_2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.x1"} : (!fir.ref<!fir.char<1,2>>, index) -> (!fir.ref<!fir.char<1,2>>, !fir.ref<!fir.char<1,2>>)
! CHECK:           %[[VAL_6:.*]] = fir.address_of(@_QFE.di.x.x1) : !fir.ref<i32>
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.di.x.x1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QFE.n.x) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] typeparams %[[VAL_1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.x"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           %[[VAL_10:.*]] = fir.address_of(@_QFEv) : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>>>
! CHECK:           %[[VAL_11:.*]] = fir.volatile_cast %[[VAL_10]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>>>) -> !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {fortran_attrs = #fir.var_attrs<allocatable, asynchronous, volatile>, uniq_name = "_QFEv"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:           %[[VAL_13:.*]] = fir.address_of(@_QFE.c.x) : !fir.ref<!fir.array<1x!fir.type<{{.*}}>>>
! CHECK:           %[[VAL_14:.*]] = fir.shape_shift %[[VAL_0]], %[[VAL_1]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.c.x"} : (!fir.ref<!fir.array<1x!fir.type<{{.*}}>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<1x!fir.type<{{.*}}>>>, !fir.ref<!fir.array<1x!fir.type<{{.*}}>>>)
! CHECK:           %[[VAL_16:.*]] = fir.address_of(@_QFE.dt.x) : !fir.ref<!fir.type<{{.*}}>>
! CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.dt.x"} : (!fir.ref<!fir.type<{{.*}}>>) -> (!fir.ref<!fir.type<{{.*}}>>, !fir.ref<!fir.type<{{.*}}>>)
! CHECK:           omp.parallel private(@_QFEv_private_class_heap_Uxrec__QFTx %[[VAL_12]]#0 -> %[[VAL_18:.*]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) {
! CHECK:             %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]] {fortran_attrs = #fir.var_attrs<allocatable, asynchronous, volatile>, uniq_name = "_QFEv"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>, volatile>, volatile>)
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
