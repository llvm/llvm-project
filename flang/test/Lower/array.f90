! RUN: bbc -o - %s | FileCheck %s

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
  ! CHECK-DAG: %[[i1:.*]] = subi %{{.*}}, %[[one:c1.*]] :
  ! CHECK: fir.load %arg4 :
  ! CHECK: %[[j1:.*]] = subi %{{.*}}, %[[one]] :
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
  ! CHECK: %[[i2:.*]] = subi %{{.*}}, %[[cc2]] :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK-DAG: %[[cc3:.*]] = fir.convert %c3{{.*}} :
  ! CHECK: %[[j2:.*]] = subi %{{.*}}, %[[cc3]] :
  ! CHECK: fir.coordinate_of %arg8, %[[i2]], %[[j2]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a3(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK-LABEL: EndIoStatement
  print *, a4(ii,jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: fir.load %arg5 :
  ! CHECK: %[[x5:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK: fir.coordinate_of %arg10, %[[x5]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a5(kk)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: %[[a6:.*]] = fir.convert %arg11 : {{.*}} -> !fir.ref<!fir.array<?xi32>>
  ! CHECK: fir.load %arg3 :
  ! CHECK-DAG: %[[x6:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK: %[[y6:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[z6:.*]] = muli %{{.}}, %[[y6]] :
  ! CHECK: %[[w6:.*]] = addi %[[z6]], %[[x6]] :
  ! CHECK: fir.coordinate_of %[[a6]], %[[w6]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a6(ii, jj)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: %[[a7:.*]] = fir.convert %arg12 : {{.*}} -> !fir.ref<!fir.array<?xf32>>
  ! CHECK: fir.load %arg5 :
  ! CHECK-DAG: %[[x7:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK-DAG: fir.load %arg4 :
  ! CHECK: %[[y7:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[z7:.*]] = muli %[[u7:.*]], %[[y7]] :
  ! CHECK: %[[w7:.*]] = addi %[[z7]], %[[x7]] :
  ! CHECK-DAG: %[[v7:.*]] = muli %[[u7]], %{{.*}} :
  ! CHECK-DAG: fir.load %arg3 :
  ! CHECK: %[[r7:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK: %[[s7:.*]] = muli %[[v7]], %[[r7]] :
  ! CHECK: %[[t7:.*]] = addi %[[s7]], %[[w7]] :
  ! CHECK: fir.coordinate_of %[[a7]], %[[t7]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a7(kk, jj, ii)
  
end subroutine s

! CHECK-LABEL range
subroutine range()
  ! Compile-time initalized arrays
  integer, dimension(10) :: a0
  real, dimension(2,3) ::  a1
  integer, dimension(3,4) :: a2
 
  a0 = (/1, 2, 3, 3, 3, 3, 3, 3, 3, 3/)
  a1 = reshape((/3.5, 3.5, 3.5, 3.5, 3.5, 3.5/), shape(a1))
  a2 = reshape((/1, 3, 3, 5, 3, 3, 3, 3, 9, 9, 9, 8/), shape(a2))
end subroutine range

! a0 array constructor
! CHECK: fir.global internal @_QQro.10xi4.{{.*}} constant : !fir.array<10xi32> {
  ! CHECK-DAG: %[[c1_i32:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[c2_i32:.*]] = constant 2 : i32
  ! CHECK-DAG: %[[c3_i32:.*]] = constant 3 : i32
  ! CHECK: %[[r1:.*]] = fir.insert_value %{{.*}}, %{{.*}}, [0 : index] :
  ! CHECK: %[[r2:.*]] = fir.insert_value %[[r1]], %{{.*}}, [1 : index] :
  ! CHECK: %[[r3:.*]] = fir.insert_on_range %[[r2]], %[[c3_i32]], [2 : index, 9 : index] :

! a1 array constructor
! CHECK: fir.global internal @_QQro.2x3xr4.{{.*}} constant : !fir.array<2x3xf32> {
  ! CHECK-DAG: %cst = constant {{.*}} : f32
  ! CHECK: %{{.*}} = fir.insert_on_range %{{[0-9]+}}, %cst, [0 : index, 1 : index, 0 : index, 2 : index] :

! a2 array constructor
! CHECK: fir.global internal @_QQro.3x4xi4.{{.*}} constant : !fir.array<3x4xi32> {
  ! CHECK-DAG: %[[c1_i32:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[c3_i32:.*]] = constant 3 : i32
  ! CHECK-DAG: %[[c5_i32:.*]] = constant 5 : i32
  ! CHECK-DAG: %[[c8_i32:.*]] = constant 8 : i32
  ! CHECK-DAG: %[[c9_i32:.*]] = constant 9 : i32
  ! CHECK: %[[r1:.*]] = fir.insert_value %{{.*}}, %{{.*}}, [0 : index, 0 : index] :
  ! CHECK: %[[r2:.*]] = fir.insert_on_range %[[r1]], %[[c3_i32]], [1 : index, 2 : index, 0 : index, 0 : index] :
  ! CHECK: %[[r3:.*]] = fir.insert_value %[[r2]], %{{.*}}, [0 : index, 1 : index] :
  ! CHECK: %[[r4:.*]] = fir.insert_on_range %[[r3]], %[[c3_i32]], [1 : index, 1 : index, 1 : index, 2 : index] :
  ! CHECK: %[[r5:.*]] = fir.insert_on_range %[[r4]], %[[c9_i32]], [2 : index, 1 : index, 2 : index, 3 : index] :
  ! CHECK: %[[r6:.*]] = fir.insert_value %[[r5]], %{{.*}}, [2 : index, 3 : index] :

! CHECK-LABEL rangeGlobal
subroutine rangeGlobal()
  ! CHECK-DAG: %[[c1_i32:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[c2_i32:.*]] = constant 2 : i32
  ! CHECK-DAG: %[[c3_i32:.*]] = constant 3 : i32
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c1_i32]], [0 : index, 1 : index] :
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c2_i32]], [2 : index, 3 : index] :
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c3_i32]], [4 : index, 5 : index] :
  integer, dimension(6) :: a0 = (/ 1, 1, 2, 2, 3, 3 /)

end subroutine rangeGlobal

block data
   ! CHECK: %[[VAL_223:.*]] = constant 1.000000e+00 : f32
   ! CHECK: %[[VAL_224:.*]] = constant 2.400000e+00 : f32
   ! CHECK: %[[VAL_225:.*]] = constant 0.000000e+00 : f32
   ! CHECK: %[[VAL_226:.*]] = fir.undefined tuple<!fir.array<5x5xf32>>
   ! CHECK: %[[VAL_227:.*]] = fir.undefined !fir.array<5x5xf32>
   ! CHECK: %[[VAL_228:.*]] = fir.insert_on_range %[[VAL_227]], %[[VAL_223]], [0 : index, 1 : index, 0 : index, 0 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_229:.*]] = fir.insert_on_range %[[VAL_228]], %[[VAL_225]], [2 : index, 4 : index, 0 : index, 0 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_230:.*]] = fir.insert_on_range %[[VAL_229]], %[[VAL_223]], [0 : index, 1 : index, 1 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_231:.*]] = fir.insert_value %[[VAL_230]], %[[VAL_225]], [2 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_232:.*]] = fir.insert_value %[[VAL_231]], %[[VAL_224]], [3 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_233:.*]] = fir.insert_value %[[VAL_232]], %[[VAL_225]], [4 : index, 1 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_234:.*]] = fir.insert_on_range %[[VAL_233]], %[[VAL_223]], [0 : index, 1 : index, 2 : index, 2 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_235:.*]] = fir.insert_value %[[VAL_234]], %[[VAL_225]], [2 : index, 2 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_236:.*]] = fir.insert_value %[[VAL_235]], %[[VAL_224]], [3 : index, 2 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_237:.*]] = fir.insert_on_range %[[VAL_236]], %[[VAL_225]], [4 : index, 2 : index, 2 : index, 3 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_238:.*]] = fir.insert_value %[[VAL_237]], %[[VAL_224]], [3 : index, 3 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_239:.*]] = fir.insert_on_range %[[VAL_238]], %[[VAL_225]], [4 : index, 4 : index, 3 : index, 4 : index] : (!fir.array<5x5xf32>, f32) -> !fir.array<5x5xf32>
   ! CHECK: %[[VAL_240:.*]] = fir.insert_value %[[VAL_226]], %[[VAL_239]], [0 : index] : (tuple<!fir.array<5x5xf32>>, !fir.array<5x5xf32>) -> tuple<!fir.array<5x5xf32>>
   ! CHECK: fir.has_value %[[VAL_240]] : tuple<!fir.array<5x5xf32>>
  real :: x(5,5)
  common /block/ x
  data x(1,1), x(2,1), x(3,1) / 1, 1, 0 /
  data x(1,2), x(2,2), x(4,2) / 1, 1, 2.4 /
  data x(1,3), x(2,3), x(4,3) / 1, 1, 2.4 /
  data x(4,4) / 2.4 /
end
