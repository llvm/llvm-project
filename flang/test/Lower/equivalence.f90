! RUN: bbc -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPs1
SUBROUTINE s1
  INTEGER i
  REAL r
  ! CHECK: = fir.alloca i8, %
  EQUIVALENCE (r,i)
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[iloc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ref<i32>
  ! CHECK-DAG: fir.store %{{.*}} to %[[iloc]] : !fir.ref<i32>
  i = 4
  ! CHECK-DAG: %[[floc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ref<f32>
  ! CHECK: %[[ld:.*]] = fir.load %[[floc]] : !fir.ref<f32>
  PRINT *, r
END SUBROUTINE s1

! CHECK-LABEL: func @_QPs2
SUBROUTINE s2
  INTEGER i(10)
  REAL r(10)
  ! CHECK: = fir.alloca i8, %
  EQUIVALENCE (r(3),i(5))
  ! CHECK: %[[iarr:.*]] = fir.convert %{{.*}} : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xi32>>
  ! CHECK: %[[ioff:.*]] = fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[farr:.*]] = fir.convert %[[ioff]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<10xf32>>
  ! CHECK: %[[ia:.*]] = fir.coordinate_of %[[iarr]], %{{.*}} : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
  ! CHECK: fir.store %{{.*}} to %[[ia]] : !fir.ref<i32>
  i(5) = 18
  ! CHECK: %[[fld:.*]] = fir.coordinate_of %[[farr]], %{{.*}} : (!fir.ref<!fir.array<10xf32>>, i64) -> !fir.ref<f32>
  ! CHECK: = fir.load %[[fld]] : !fir.ref<f32>
  PRINT *, r(3)
END SUBROUTINE s2
