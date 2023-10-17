! RUN: %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: func.func @_QPprivate_common() {
!CHECK: omp.parallel {
!CHECK: %[[X:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFprivate_commonEx"}
!CHECK: %[[Y:.*]] = fir.alloca f32 {bindc_name = "y", pinned, uniq_name = "_QFprivate_commonEy"}
!CHECK: omp.terminator
!CHECK: }
!CHECK: return
!CHECK: }
subroutine private_common
  common /c/ x, y
  real x, y
  !$omp parallel private(/c/)
  !$omp end parallel
end subroutine

!CHECK: %[[val_0:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<74xi8>>
!CHECK: %[[val_1:.*]] = fir.convert %0 : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_2:.*]] = fir.coordinate_of %[[val_1]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_3:.*]] = fir.convert %[[val_2]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_4:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c4:.*]] = arith.constant 4 : index
!CHECK: %[[val_5:.*]] = fir.coordinate_of %[[val_4]], %[[val_c4]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_6:.*]] = fir.convert %[[val_5]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
!CHECK: %[[val_7:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c44:.*]] = arith.constant 44 : index
!CHECK: %[[val_8:.*]] = fir.coordinate_of %[[val_7]], %[[val_c44]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_9:.*]] = fir.convert %[[val_8]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
!CHECK: %[[val_c5:.*]] = arith.constant 5 : index
!CHECK: %[[val_10:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c49:.*]] = arith.constant 49 : index
!CHECK: %[[val_11:.*]] = fir.coordinate_of %[[val_10]], %[[val_c49]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_12:.*]] = fir.convert %[[val_11]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK: %[[val_c5_0:.*]] = arith.constant 5 : index
!CHECK: %[[val_14:.*]] = fir.emboxchar %[[val_9]], %[[val_c5]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK: %[[val_15:.*]] = fir.convert %[[val_12]] : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,?>>
!CHECK: %[[val_16:.*]] = fir.emboxchar %[[val_15]], %[[val_c5_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
!CHECK: fir.call @_QPsub1(%[[val_3]], %[[val_6]], %[[val_14]], %[[val_16]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.array<10xf32>>, !fir.boxchar<1>, !fir.boxchar<1>) -> ()
!CHECK: omp.parallel {
!CHECK: %[[val_21:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFprivate_clause_commonblockEa"}
!CHECK: %[[val_22:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "b", pinned, uniq_name = "_QFprivate_clause_commonblockEb"}
!CHECK: %[[val_23:.*]] = fir.alloca !fir.char<1,5> {bindc_name = "c", pinned, uniq_name = "_QFprivate_clause_commonblockEc"}
!CHECK: %[[val_24:.*]] = fir.alloca !fir.array<5x!fir.char<1,5>> {bindc_name = "d", pinned, uniq_name = "_QFprivate_clause_commonblockEd"}
!CHECK: %[[val_26:.*]] = fir.emboxchar %[[val_23]], %[[val_c5]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK: %[[val_27:.*]] = fir.convert %[[val_24]] : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,?>>
!CHECK: %[[val_28:.*]] = fir.emboxchar %[[val_27]], %[[val_c5_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
!CHECK: fir.call @_QPsub2(%[[val_21]], %[[val_22]], %[[val_26]], %[[val_28]]) fastmath<contract> : (!fir.ref<i32>, !fir.ref<!fir.array<10xf32>>, !fir.boxchar<1>, !fir.boxchar<1>) -> ()
!CHECK: omp.terminator
!CHECK: }
!CHECK: %[[val_18:.*]] = fir.emboxchar %[[val_9]], %[[val_c5]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
!CHECK: %[[val_19:.*]] = fir.convert %[[val_12]] : (!fir.ref<!fir.array<5x!fir.char<1,5>>>) -> !fir.ref<!fir.char<1,?>>
!CHECK: %[[val_20:.*]] = fir.emboxchar %[[val_19]], %[[val_c5_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
!CHECK: fir.call @_QPsub3(%[[val_3]], %[[val_6]], %[[val_18]], %[[val_20]]) fastmath<contract> : {{.*}}
!CHECK: return
!CHECK: }
subroutine private_clause_commonblock()
  integer::a
  real::b(10)
  character(5):: c, d(5)
  common /blk/ a, b, c, d
  
  call sub1(a, b, c, d)
  !$omp parallel private(/blk/)
        call sub2(a, b, c, d)
  !$omp end parallel
  call sub3(a, b, c, d)
end subroutine

!CHECK: func.func @_QPprivate_clause_commonblock_pointer() {
!CHECK: %[[val_0:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<74xi8>>
!CHECK: %[[val_1:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c24:.*]] = arith.constant 24 : index
!CHECK: %[[val_2:.*]] = fir.coordinate_of %[[val_1]], %[[val_c24]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_3:.*]] = fir.convert %[[val_2]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[val_4:.*]] = fir.convert %[[val_0]] : (!fir.ref<!fir.array<74xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[val_c0:.*]] = arith.constant 0 : index
!CHECK: %[[val_5:.*]] = fir.coordinate_of %[[val_4]], %[[val_c0]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[val_6:.*]] = fir.convert %[[val_5]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK: %[[val_7:.*]] = fir.load %[[val_6]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK: %[[val_8:.*]] = fir.box_addr %[[val_7]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK: %[[val_9:.*]] = fir.convert %[[val_8]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK: fir.call @_QPsub4(%[[val_9]], %[[val_3]]) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK: omp.parallel {
!CHECK: %[[val_13:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.complex<4>>> {bindc_name = "c", pinned, uniq_name = "_QFprivate_clause_commonblock_pointerEc"}
!CHECK: %[[val_14:.*]] = fir.alloca i32 {bindc_name = "a", pinned, uniq_name = "_QFprivate_clause_commonblock_pointerEa"}
!CHECK: %[[val_15:.*]] = fir.load %[[val_13]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK: %[[val_16:.*]] = fir.box_addr %[[val_15]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK: %[[val_17:.*]] = fir.convert %[[val_16]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK: fir.call @_QPsub5(%[[val_17]], %[[val_14]]) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK: omp.terminator
!CHECK: }
!CHECK: %[[val_10:.*]] = fir.load %[[val_6]] : !fir.ref<!fir.box<!fir.ptr<!fir.complex<4>>>>
!CHECK: %[[val_11:.*]] = fir.box_addr %[[val_10]] : (!fir.box<!fir.ptr<!fir.complex<4>>>) -> !fir.ptr<!fir.complex<4>>
!CHECK: %[[val_12:.*]] = fir.convert %[[val_11]] : (!fir.ptr<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>>
!CHECK: fir.call @_QPsub6(%[[val_12]], %[[val_3]]) fastmath<contract> : (!fir.ref<!fir.complex<4>>, !fir.ref<i32>) -> ()
!CHECK: return
!CHECK: }
subroutine private_clause_commonblock_pointer()
  complex, pointer :: c
  integer:: a
  common /blk/ c, a
  call sub4(c, a)
  !$omp parallel private(/blk/)
        call sub5(c, a)
  !$omp end parallel
  call sub6(c, a)
end subroutine
