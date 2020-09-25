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
  ! CHECK-DAG: %[[c3_i32:.*]] = constant 3 : i32
  ! CHECK-DAG: %[[c9_i32:.*]] = constant 9 : i32
  ! CHECK-DAG: %[[c0:.*]] = constant 0 : index
  ! CHECK-DAG: %[[c1:.*]] = constant 1 : index
  ! CHECK-DAG: %[[c2:.*]] = constant 2 : index
  ! CHECK-DAG: %[[c3:.*]] = constant 3 : index
  ! CHECK-DAG: %[[c9:.*]] = constant 9 : index
  integer, dimension(10) :: a0
  real, dimension(2,3) ::  a1
  integer, dimension(3,4) :: a2
 
  ! CHECK: %[[r1:.*]] = fir.insert_value %{{.*}}, %{{.*}}, %{{.*}} :
  ! CHECK: %[[r2:.*]] = fir.insert_value %[[r1]], %{{.*}}, %{{.*}} :
  ! CHECK: %[[r3:.*]] = fir.insert_on_range %[[r2]], %[[c3_i32]], %[[c2]], %[[c9]] :
  a0 = (/1, 2, 3, 3, 3, 3, 3, 3, 3, 3/)
  ! CHECK: %{{.*}} = fir.insert_on_range %{{[0-9]+}}, %{{.*}}, %[[c0]], %[[c1]], %[[c0]], %[[c2]] :
  a1 = reshape((/3.5, 3.5, 3.5, 3.5, 3.5, 3.5/), shape(a1))
  ! CHECK: %[[r4:.*]] = fir.insert_value %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} :
  ! CHECK: %[[r5:.*]] = fir.insert_on_range %[[r4]], %[[c3_i32]], %[[c1]], %[[c2]], %[[c0]], %[[c0]] :
  ! CHECK: %[[r6:.*]] = fir.insert_value %[[r5]], %{{.*}}, %{{.*}}, %{{.*}} :
  ! CHECK: %[[r7:.*]] = fir.insert_on_range %[[r6]], %[[c3_i32]], %[[c1]], %[[c1]], %[[c1]], %[[c2]] :
  ! CHECK: %[[r8:.*]] = fir.insert_on_range %[[r7]], %[[c9_i32]], %[[c2]], %[[c1]], %[[c2]], %[[c3]] :
  ! CHECK: %[[r9:.*]] = fir.insert_value %[[r8]], %{{.*}}, %{{.*}}, %{{.*}} :
  a2 = reshape((/1, 3, 3, 5, 3, 3, 3, 3, 9, 9, 9, 8/), shape(a2))

end subroutine range

! CHECK-LABEL rangeGlobal
subroutine rangeGlobal()
  ! CHECK-DAG: %[[c1_i32:.*]] = constant 1 : i32
  ! CHECK-DAG: %[[c2_i32:.*]] = constant 2 : i32
  ! CHECK-DAG: %[[c3_i32:.*]] = constant 3 : i32
  ! CHECK-DAG: %[[c0:.*]] = constant 0 : index
  ! CHECK-DAG: %[[c1:.*]] = constant 1 : index
  ! CHECK-DAG: %[[c2:.*]] = constant 2 : index
  ! CHECK-DAG: %[[c3:.*]] = constant 3 : index
  ! CHECK-DAG: %[[c4:.*]] = constant 4 : index
  ! CHECK-DAG: %[[c5:.*]] = constant 5 : index
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c1_i32]], %[[c0]], %[[c1]] :
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c2_i32]], %[[c2]], %[[c3]] :
  ! CHECK: %{{.*}} = fir.insert_on_range %{{.*}}, %[[c3_i32]], %[[c4]], %[[c5]] :
  integer, dimension(6) :: a0 = (/ 1, 1, 2, 2, 3, 3 /)

end subroutine rangeGlobal

block data
  real :: x(5,5)
  common /block/ x
  data x(1,1), x(2,1), x(3,1) / 1, 1, 0 /
  data x(1,2), x(2,2), x(4,2) / 1, 1, 2.4 /
  data x(1,3), x(2,3), x(4,3) / 1, 1, 2.4 /
  data x(4,4) / 2.4 /
end
