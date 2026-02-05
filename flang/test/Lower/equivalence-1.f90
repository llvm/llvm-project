! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPs1()
SUBROUTINE s1
  INTEGER i
  REAL r
  ! CHECK: %[[group:.*]] = fir.alloca !fir.array<4xi8> {uniq_name = "_QFs1Ei"}
  EQUIVALENCE (r,i)
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[group]], %c0{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[iloc:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<i32>
  ! CHECK: %[[i_decl:.*]]:2 = hlfir.declare %[[iloc]] storage(%[[group]][0]) {uniq_name = "_QFs1Ei"}
  ! CHECK: %[[coor2:.*]] = fir.coordinate_of %[[group]], %c0{{.*}} : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[rloc:.*]] = fir.convert %[[coor2]] : (!fir.ref<i8>) -> !fir.ptr<f32>
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %[[rloc]] storage(%[[group]][0]) {uniq_name = "_QFs1Er"}
  i = 4
  ! CHECK: hlfir.assign %c4{{.*}} to %[[i_decl]]#0
  PRINT *, r
  ! CHECK: fir.load %[[r_decl]]#0
END SUBROUTINE s1

! CHECK-LABEL: func.func @_QPs2()
SUBROUTINE s2
  INTEGER i(10)
  REAL r(10)
  ! CHECK: %[[arr:.*]] = fir.alloca !fir.array<48xi8>
  EQUIVALENCE (r(3),i(5))
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[arr]], %c0{{.*}} : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[iarr:.*]] = fir.convert %[[coor]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xi32>>
  ! CHECK: %[[i_decl:.*]]:2 = hlfir.declare %[[iarr]](%{{.*}}) storage(%[[arr]][0]) {uniq_name = "_QFs2Ei"}
  ! CHECK: %[[coor2:.*]] = fir.coordinate_of %[[arr]], %c8{{.*}} : (!fir.ref<!fir.array<48xi8>>, index) -> !fir.ref<i8>
  ! CHECK: %[[rarr:.*]] = fir.convert %[[coor2]] : (!fir.ref<i8>) -> !fir.ptr<!fir.array<10xf32>>
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %[[rarr]](%{{.*}}) storage(%[[arr]][8]) {uniq_name = "_QFs2Er"}
  i(5) = 18
  ! CHECK: %[[ides:.*]] = hlfir.designate %[[i_decl]]#0 (%c5{{.*}})
  ! CHECK: hlfir.assign %c18{{.*}} to %[[ides]]
  PRINT *, r(3)
  ! CHECK: %[[rdes:.*]] = hlfir.designate %[[r_decl]]#0 (%c3{{.*}})
  ! CHECK: fir.load %[[rdes]]
END SUBROUTINE s2

! CHECK-LABEL: func.func @_QPs3()
SUBROUTINE s3
  REAL r(10)
  TYPE t
    SEQUENCE
    REAL r(10)
  END TYPE t
  TYPE(t) x
  ! CHECK: %[[group:.*]] = fir.alloca !fir.array<40xi8>
  EQUIVALENCE (r,x)
  ! CHECK: %[[r_decl:.*]]:2 = hlfir.declare %{{.*}} storage(%[[group]][0]) {uniq_name = "_QFs3Er"}
  ! CHECK: %[[x_decl:.*]]:2 = hlfir.declare %{{.*}} storage(%[[group]][0]) {uniq_name = "_QFs3Ex"}
  x%r(9) = 9.0
  ! CHECK: %[[xdes:.*]] = hlfir.designate %[[x_decl]]#0{"r"} <%{{.*}}> (%c9{{.*}})
  ! CHECK: hlfir.assign %{{.*}} to %[[xdes]]
  PRINT *, r(9)
  ! CHECK: %[[rdes:.*]] = hlfir.designate %[[r_decl]]#0 (%c9{{.*}})
  ! CHECK: fir.load %[[rdes]]
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
