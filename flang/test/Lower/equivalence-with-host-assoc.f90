! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s --check-prefixes=FIR
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s --check-prefixes=HLFIR

subroutine test1()
  integer :: i1 = 1
  integer :: j1
  equivalence(i1,j1)
contains
  subroutine inner
    i1 = j1
  end subroutine inner
end subroutine test1
! FIR-LABEL:   func.func private @_QFtest1Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! FIR:           %[[VAL_0:.*]] = fir.address_of(@_QFtest1Ei1) : !fir.ref<!fir.array<1xi32>>
! FIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<1xi32>>) -> !fir.ref<!fir.array<4xi8>>
! FIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_5:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_5]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ptr<i32>
! FIR:           fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ptr<i32>
! FIR:           return
! FIR:         }

! HLFIR-LABEL:   func.func private @_QFtest1Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_0:.*]] = fir.address_of(@_QFtest1Ei1) : !fir.ref<!fir.array<1xi32>>
! HLFIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<1xi32>>) -> !fir.ref<!fir.array<4xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest1Ei1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_6:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QFtest1Ej1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_10:.*]] = fir.load %[[VAL_9]]#0 : !fir.ptr<i32>
! HLFIR:           hlfir.assign %[[VAL_10]] to %[[VAL_5]]#0 : i32, !fir.ptr<i32>
! HLFIR:           return
! HLFIR:         }

module test2
  real :: f1, f2
  equivalence(f1, f2)
contains
  subroutine host
    real :: f1 = 1
    real :: f2
    equivalence(f1, f2)
  contains
    subroutine inner
      f1 = f2
    end subroutine inner
  end subroutine host
end module test2
! FIR-LABEL:   func.func private @_QMtest2FhostPinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! FIR:           %[[VAL_0:.*]] = fir.address_of(@_QMtest2FhostEf1) : !fir.ref<!fir.array<1xi32>>
! FIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<1xi32>>) -> !fir.ref<!fir.array<4xi8>>
! FIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<f32>
! FIR:           %[[VAL_5:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_5]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<i8>) -> !fir.ptr<f32>
! FIR:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ptr<f32>
! FIR:           fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ptr<f32>
! FIR:           return
! FIR:         }

! HLFIR-LABEL:   func.func private @_QMtest2FhostPinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_0:.*]] = fir.address_of(@_QMtest2FhostEf1) : !fir.ref<!fir.array<1xi32>>
! HLFIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.array<1xi32>>) -> !fir.ref<!fir.array<4xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<f32>
! HLFIR:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QMtest2FhostEf1"} : (!fir.ptr<f32>) -> (!fir.ptr<f32>, !fir.ptr<f32>)
! HLFIR:           %[[VAL_6:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_6]] : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ptr<f32>
! HLFIR:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] {uniq_name = "_QMtest2FhostEf2"} : (!fir.ptr<f32>) -> (!fir.ptr<f32>, !fir.ptr<f32>)
! HLFIR:           %[[VAL_19:.*]] = fir.load %[[VAL_9]]#0 : !fir.ptr<f32>
! HLFIR:           hlfir.assign %[[VAL_19]] to %[[VAL_5]]#0 : f32, !fir.ptr<f32>
! HLFIR:           return
! HLFIR:         }

subroutine test3()
  integer :: i1 = 1
  integer :: j1, k1
  common /blk/ k1
  equivalence(i1,j1,k1)
contains
  subroutine inner
    i1 = j1 + k1
  end subroutine inner
end subroutine test3
! FIR-LABEL:   func.func private @_QFtest3Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! FIR:           %[[VAL_0:.*]] = fir.address_of(@blk_) : !fir.ref<tuple<i32>>
! FIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_6:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_6]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_10:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_10]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_13:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
! FIR:           %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ptr<i32>
! FIR:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
! FIR:           fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ptr<i32>
! FIR:           return
! FIR:         }

! HLFIR-LABEL:   func.func private @_QFtest3Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_0:.*]] = fir.address_of(@blk_) : !fir.ref<tuple<i32>>
! HLFIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest3Ei1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_7:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_6]], %[[VAL_7]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFtest3Ej1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_12:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_11]], %[[VAL_12]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFtest3Ek1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_16:.*]] = fir.load %[[VAL_10]]#0 : !fir.ptr<i32>
! HLFIR:           %[[VAL_17:.*]] = fir.load %[[VAL_15]]#0 : !fir.ptr<i32>
! HLFIR:           %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
! HLFIR:           hlfir.assign %[[VAL_18]] to %[[VAL_5]]#0 : i32, !fir.ptr<i32>
! HLFIR:           return
! HLFIR:         }

subroutine test4()
  integer :: i1
  integer :: j1, k1
  common /blk/ k1
  equivalence(i1,j1,k1)
contains
  subroutine inner
    i1 = j1 + k1
  end subroutine inner
end subroutine test4
! FIR-LABEL:   func.func private @_QFtest4Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! FIR:           %[[VAL_0:.*]] = fir.address_of(@blk_) : !fir.ref<tuple<i32>>
! FIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_5:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_6:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_6]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR:           %[[VAL_10:.*]] = arith.constant 0 : index
! FIR:           %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_10]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! FIR:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! FIR:           %[[VAL_13:.*]] = fir.load %[[VAL_8]] : !fir.ptr<i32>
! FIR:           %[[VAL_14:.*]] = fir.load %[[VAL_12]] : !fir.ptr<i32>
! FIR:           %[[VAL_15:.*]] = arith.addi %[[VAL_13]], %[[VAL_14]] : i32
! FIR:           fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ptr<i32>
! FIR:           return
! FIR:         }

! HLFIR-LABEL:   func.func private @_QFtest4Pinner() attributes {fir.host_symbol = {{.*}}, llvm.linkage = #llvm.linkage<internal>} {
! HLFIR:           %[[VAL_0:.*]] = fir.address_of(@blk_) : !fir.ref<tuple<i32>>
! HLFIR:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_2:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFtest4Ei1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_6:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_7:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_6]], %[[VAL_7]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] {uniq_name = "_QFtest4Ej1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<tuple<i32>>) -> !fir.ref<!fir.array<?xi8>>
! HLFIR:           %[[VAL_12:.*]] = arith.constant 0 : index
! HLFIR:           %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_11]], %[[VAL_12]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
! HLFIR:           %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<i8>) -> !fir.ptr<i32>
! HLFIR:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_14]] {uniq_name = "_QFtest4Ek1"} : (!fir.ptr<i32>) -> (!fir.ptr<i32>, !fir.ptr<i32>)
! HLFIR:           %[[VAL_16:.*]] = fir.load %[[VAL_10]]#0 : !fir.ptr<i32>
! HLFIR:           %[[VAL_17:.*]] = fir.load %[[VAL_15]]#0 : !fir.ptr<i32>
! HLFIR:           %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
! HLFIR:           hlfir.assign %[[VAL_18]] to %[[VAL_5]]#0 : i32, !fir.ptr<i32>
! HLFIR:           return
! HLFIR:         }
