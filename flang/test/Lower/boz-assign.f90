! RUN: bbc -emit-fir -o - %s | FileCheck %s

subroutine boz1
!CHECK-LABEL: func.func @_QPboz1
!CHECK: %[[CONST3:.*]] = arith.constant 243 : i128
!CHECK: %[[IDX2:.*]] = arith.constant 2 : index  
!CHECK: %[[CONST2:.*]] = arith.constant 42 : i128
!CHECK: %[[IDX1:.*]] = arith.constant 1 : index  
!CHECK: %[[CONST1:.*]] = arith.constant 128 : i128
!CHECK: %[[IDX3:.*]] = arith.constant 3 : index  
  INTEGER :: a(3)
!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<3xi32>
!CHECK: %[[ARRAY:.*]] = fir.declare %[[ALLOCA]]({{.*}}) {{.*}} : (!fir.ref<!fir.array<3xi32>>, {{.*}}) -> !fir.ref<!fir.array<3xi32>>
  a(1)=B'10000000'
!CHECK: %[[CVT1:.*]] = fir.convert %[[CONST1]] : (i128) -> i32
!CHECK: %[[ARR_IDX1:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX1]]
!CHECK: fir.store %[[CVT1]] to %[[ARR_IDX1]] : !fir.ref<i32>
  a(2)=O'52'
!CHECK: %[[CVT2:.*]] = fir.convert %[[CONST2]] : (i128) -> i32
!CHECK: %[[ARR_IDX2:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX2]]
!CHECK: fir.store %[[CVT2]] to %[[ARR_IDX2]] : !fir.ref<i32>
  a(3)=Z'F3'
!CHECK: %[[CVT3:.*]] = fir.convert %[[CONST3]] : (i128) -> i32
!CHECK: %[[ARR_IDX3:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX3]]
!CHECK: fir.store %[[CVT3]] to %[[ARR_IDX3]] : !fir.ref<i32>
end subroutine boz1

subroutine boz2
!CHECK-LABEL: func.func @_QPboz2
!CHECK: %[[CONST3:.*]] = arith.constant 8070450532247928832 : i128
!CHECK: %[[IDX2:.*]] = arith.constant 2 : index  
!CHECK: %[[CONST2:.*]] = arith.constant 68719476736  : i128
!CHECK: %[[IDX1:.*]] = arith.constant 1 : index  
!CHECK: %[[CONST1:.*]] = arith.constant 9223372036854775808 : i128
!CHECK: %[[IDX3:.*]] = arith.constant 3 : index  
  INTEGER(8) :: b(3)
!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<3xi64>
!CHECK: %[[ARRAY:.*]] = fir.declare %[[ALLOCA]]({{.*}}) {{.*}} : (!fir.ref<!fir.array<3xi64>>, {{.*}}) -> !fir.ref<!fir.array<3xi64>>
  b(1)=B'1000000000000000000000000000000000000000000000000000000000000000'
!CHECK: %[[CVT1:.*]] = fir.convert %[[CONST1]] : (i128) -> i64
!CHECK: %[[ARR_IDX1:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX1]]
!CHECK: fir.store %[[CVT1]] to %[[ARR_IDX1]] : !fir.ref<i64>
  b(2)=O'01000000000000'
!CHECK: %[[CVT2:.*]] = fir.convert %[[CONST2]] : (i128) -> i64
!CHECK: %[[ARR_IDX2:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX2]]
!CHECK: fir.store %[[CVT2]] to %[[ARR_IDX2]] : !fir.ref<i64>
  b(3)=Z'7000000000000000'
!CHECK: %[[CVT3:.*]] = fir.convert %[[CONST3]] : (i128) -> i64
!CHECK: %[[ARR_IDX3:.*]] = fir.array_coor %[[ARRAY]]({{.*}}) %[[IDX3]]
!CHECK: fir.store %[[CVT3]] to %[[ARR_IDX3]] : !fir.ref<i64>
end subroutine boz2

subroutine boz3
!CHECK-LABEL: func.func @_QPboz3
  INTEGER(16) :: c
!CHECK: %[[CONST:.*]] = arith.constant 158798437896437949616241483468158498679 : i128
!CHECK: %[[ALLOCA:.*]] = fir.alloca i128
!CHECK: %[[VAR:.*]] = fir.declare %[[ALLOCA]] {{.*}} : (!fir.ref<i128>) -> !fir.ref<i128>
  c = Z'77777777777777777777777777777777'
!CHECK: fir.store %[[CONST]] to %[[VAR]] : !fir.ref<i128>
end
