! RUN: bbc -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: fir.global @block_
! CHECK-DAG: %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f32
! CHECK-DAG: %[[VAL_2:.*]] = arith.constant 2.400000e+00 : f32
! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
! CHECK: %[[VAL_4:.*]] = fir.zero_bits tuple<!fir.array<5x5xf32>>
! CHECK: %[[VAL_5:.*]] = fir.undefined !fir.array<5x5xf32>
! CHECK: %[[VAL_6:.*]] = fir.insert_on_range %[[VAL_5]], %[[VAL_1]] from (0, 0) to (1, 0) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_7:.*]] = fir.insert_on_range %[[VAL_6]], %[[VAL_3]] from (2, 0) to (4, 0) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_8:.*]] = fir.insert_on_range %[[VAL_7]], %[[VAL_1]] from (0, 1) to (1, 1) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_3]], [2 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_2]], [3 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_11:.*]] = fir.insert_value %[[VAL_10]], %[[VAL_3]], [4 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_12:.*]] = fir.insert_on_range %[[VAL_11]], %[[VAL_1]] from (0, 2) to (1, 2) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_3]], [2 : index, 2 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_2]], [3 : index, 2 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_15:.*]] = fir.insert_on_range %[[VAL_14]], %[[VAL_3]] from (4, 2) to (2, 3) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_16:.*]] = fir.insert_value %[[VAL_15]], %[[VAL_2]], [3 : index, 3 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_17:.*]] = fir.insert_on_range %[[VAL_16]], %[[VAL_3]] from (4, 3) to (4, 4) : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
! CHECK: %[[VAL_18:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_17]], [0 : index] : (tuple<!fir.array<5x5xf32>>, !fir.array<5x5xf32>) -> tuple<!fir.array<5x5xf32>>
! CHECK: fir.has_value %[[VAL_18]] : tuple<!fir.array<5x5xf32>>

subroutine s(i,j,k,ii,jj,kk,a1,a2,a3,a4,a5,a6,a7)
  integer i, j, k, ii, jj, kk

  ! extents are compile-time constant
  real a1(10,20)
  integer a2(30,*)
  real a3(2:40,3:50)
  integer a4(4:60, 5:*)

  ! extents computed at run-time
  real a5(i:j)
  integer a6(6:i,j:*)
  real a7(i:70,7:j,k:80)

  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK-DAG: fir.load %arg3 :
  ! CHECK-DAG: %[[i1:.*]] = arith.subi %{{.*}}, %[[one:c1.*]] :
  ! CHECK: fir.load %arg4 :
  ! CHECK: %[[j1:.*]] = arith.subi %{{.*}}, %[[one]] :
  ! CHECK: fir.coordinate_of %arg6, %[[i1]], %[[j1]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a1(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: fir.coordinate_of %{{[0-9]+}}, %{{[0-9]+}} : {{.*}} -> !fir.ref<i32>
  ! CHECK-LABEL: EndIoStatement
  print *, a2(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK-DAG: fir.load %arg3 :
  ! CHECK-DAG: %[[cc2:.*]] = fir.convert %c2{{.*}} :
  ! CHECK: %[[i2:.*]] = arith.subi %{{.*}}, %[[cc2]] :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK-DAG: %[[cc3:.*]] = fir.convert %c3{{.*}} :
  ! CHECK: %[[j2:.*]] = arith.subi %{{.*}}, %[[cc3]] :
  ! CHECK: fir.coordinate_of %arg8, %[[i2]], %[[j2]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a3(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK-LABEL: EndIoStatement
  print *, a4(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: fir.load %arg5 :
  ! CHECK: %[[x5:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK: fir.coordinate_of %arg10, %[[x5]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a5(kk)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: %[[a6:.*]] = fir.convert %arg11 : {{.*}} -> !fir.ref<!fir.array<?xi32>>
  ! CHECK: fir.load %arg3 :
  ! CHECK-DAG: %[[x6:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK: %[[y6:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[z6:.*]] = arith.muli %{{.}}, %[[y6]] :
  ! CHECK: %[[w6:.*]] = arith.addi %[[z6]], %[[x6]] :
  ! CHECK: fir.coordinate_of %[[a6]], %[[w6]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a6(ii, jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: %[[a7:.*]] = fir.convert %arg12 : {{.*}} -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: fir.load %arg5 :
  ! CHECK-DAG: %[[x7:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK: %[[y7:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[z7:.*]] = arith.muli %[[u7:.*]], %[[y7]] :
  ! CHECK: %[[w7:.*]] = arith.addi %[[z7]], %[[x7]] :
  ! CHECK-DAG: %[[v7:.*]] = arith.muli %[[u7]], %{{.*}} :
  ! CHECK-DAG: fir.load %arg3 :
  ! CHECK: %[[r7:.*]] = arith.subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[s7:.*]] = arith.muli %[[v7]], %[[r7]] :
  ! CHECK: %[[t7:.*]] = arith.addi %[[s7]], %[[w7]] :
  ! CHECK: fir.coordinate_of %[[a7]], %[[t7]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a7(kk, jj, ii)
  
end subroutine s

! CHECK-LABEL: range
subroutine range()
  ! Compile-time initalized arrays
  integer, dimension(10) :: a0
  real, dimension(2,3) ::  a1
  integer, dimension(3,4) :: a2
  integer, dimension(2,3,4) :: a3
  complex, dimension(2,3) :: c0, c1

  a0 = (/1, 2, 3, 3, 3, 3, 3, 3, 3, 3/)
  a1 = reshape((/3.5, 3.5, 3.5, 3.5, 3.5, 3.5/), shape(a1))
  a2 = reshape((/1, 3, 3, 5, 3, 3, 3, 3, 9, 9, 9, 8/), shape(a2))
  a3 = reshape((/1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12/), shape(a3))

  c0 = reshape((/(1.0, 1.5), (2.0, 2.5), (3.0, 3.5), (4.0, 4.5), (5.0, 5.5), (6.0, 6.5)/), shape(c0))
  data c1/6 * (0.0, 0.0)/
end subroutine range

! c1 data
! CHECK: fir.global internal @_QFrangeEc1(dense<(0.000000e+00,0.000000e+00)> : tensor<3x2xcomplex<f32>>) : !fir.array<2x3xcomplex<f32>>

! a0 array constructor
! CHECK: fir.global internal @_QQro.10xi4.{{.*}}(dense<[1, 2, 3, 3, 3, 3, 3, 3, 3, 3]> : tensor<10xi32>) constant : !fir.array<10xi32>

! a1 array constructor
! CHECK: fir.global internal @_QQro.2x3xr4.{{.*}}(dense<3.500000e+00> : tensor<3x2xf32>) constant : !fir.array<2x3xf32>

! a2 array constructor
! CHECK: fir.global internal @_QQro.3x4xi4.{{.*}}(dense<{{\[\[1, 3, 3], \[5, 3, 3], \[3, 3, 9], \[9, 9, 8]]}}> : tensor<4x3xi32>) constant : !fir.array<3x4xi32>

! a3 array constructor
! CHECK: fir.global internal @_QQro.2x3x4xi4.{{.*}}(dense<{{\[\[\[1, 1], \[2, 2], \[3, 3]], \[\[4, 4], \[5, 5], \[6, 6]], \[\[7, 7], \[8, 8], \[9, 9]], \[\[10, 10], \[11, 11], \[12, 12]]]}}> : tensor<4x3x2xi32>) constant : !fir.array<2x3x4xi32>

! c0 array constructor
! CHECK: fir.global internal @_QQro.2x3xz4.{{.*}}(dense<{{\[}}[(1.000000e+00,1.500000e+00), (2.000000e+00,2.500000e+00)], [(3.000000e+00,3.500000e+00), (4.000000e+00,4.500000e+00)], [(5.000000e+00,5.500000e+00), (6.000000e+00,6.500000e+00)]]> : tensor<3x2xcomplex<f32>>) constant : !fir.array<2x3xcomplex<f32>>

! CHECK-LABEL: rangeGlobal
subroutine rangeGlobal()
! CHECK: fir.global internal @_QFrangeglobal{{.*}}(dense<[1, 1, 2, 2, 3, 3]> : tensor<6xi32>) : !fir.array<6xi32>
  integer, dimension(6) :: a0 = (/ 1, 1, 2, 2, 3, 3 /)

end subroutine rangeGlobal

! CHECK-LABEL: hugeGlobal
subroutine hugeGlobal()
  integer, parameter :: D = 500
  integer, dimension(D, D) :: a

! CHECK: fir.global internal @_QQro.500x500xi4.{{.*}}(dense<{{.*}}> : tensor<500x500xi32>) constant : !fir.array<500x500xi32>
  a = reshape((/(i, i = 1, D * D)/), shape(a))
end subroutine hugeGlobal

block data
  real(selected_real_kind(6)) :: x(5,5)
  common /block/ x
  data x(1,1), x(2,1), x(3,1) / 1, 1, 0 /
  data x(1,2), x(2,2), x(4,2) / 1, 1, 2.4 /
  data x(1,3), x(2,3), x(4,3) / 1, 1, 2.4 /
  data x(4,4) / 2.4 /
end
