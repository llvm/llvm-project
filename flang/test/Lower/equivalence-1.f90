! RUN: bbc -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPs1
SUBROUTINE s1
  INTEGER i
  REAL r
  ! CHECK: = fir.alloca !fir.array<4xi8> {uniq_name = "_QFs1Ei"}
  EQUIVALENCE (r,i)
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[iloc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<i32>
  ! CHECK-DAG: fir.store %{{.*}} to %[[iloc]] : !fir.ptr<i32>
  i = 4
  ! CHECK-DAG: %[[floc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<f32>
  ! CHECK: %[[ld:.*]] = fir.load %[[floc]] : !fir.ptr<f32>
  PRINT *, r
END SUBROUTINE s1

! CHECK-LABEL: func @_QPs2
SUBROUTINE s2
  INTEGER i(10)
  REAL r(10)
  ! CHECK: %[[arr:.*]] = fir.alloca !fir.array<48xi8>
  EQUIVALENCE (r(3),i(5))
  ! CHECK: %[[iarr:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
  ! CHECK: %[[foff:.*]] = fir.coordinate_of %[[arr]], %{{.*}} : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[farr:.*]] = fir.convert %[[foff]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
  ! CHECK: %[[ia:.*]] = fir.coordinate_of %[[iarr]], %{{.*}} : (!fir.ptr<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
  ! CHECK: fir.store %{{.*}} to %[[ia]] : !fir.ref<i32>
  i(5) = 18
  ! CHECK: %[[fld:.*]] = fir.coordinate_of %[[farr]], %{{.*}} : (!fir.ptr<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: = fir.load %[[fld]] : !fir.ref<f32>
  PRINT *, r(3)
END SUBROUTINE s2

! CHECK-LABEL: func @_QPs3
SUBROUTINE s3
  REAL r(10)
  TYPE t
    SEQUENCE
    REAL r(10)
  END TYPE t
  TYPE(t) x
  ! CHECK: %[[group:.*]] = fir.alloca !fir.array<40xi8>
  EQUIVALENCE (r,x)
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[group]], %c0 : (!fir.ref<!fir.array<40xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[rloc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
  ! CHECK: %[[xloc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.type<_QFs3Tt{r:!fir.array<10xf32>}>>
  ! CHECK: %[[fidx:.*]] = fir.field_index r, !fir.type<_QFs3Tt{r:!fir.array<10xf32>}>
  ! CHECK: %[[xrloc:.*]] = fir.coordinate_of %[[xloc]], %[[fidx]] :
  ! CHECK: %[[v1loc:.*]] = fir.coordinate_of %[[xrloc]], %c8_i64 : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: fir.store %{{.*}} to %[[v1loc]] : !fir.ref<f32>
  x%r(9) = 9.0
  ! CHECK: %[[v2loc:.*]] = fir.coordinate_of %[[rloc]], %c8_i64 : (!fir.ptr<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: %{{.*}} = fir.load %[[v2loc]] : !fir.ref<f32>
  PRINT *, r(9)
END SUBROUTINE s3
  
! test that equivalence in main program containing arrays are placed in global memory.
! CHECK: fir.global internal @_QFEa : !fir.array<400000000xi8>
  integer :: a, b(100000000)
  equivalence (a, b)
  b(1) = 42
  print *, a

  CALL s1
  CALL s2
  CALL s3
END
