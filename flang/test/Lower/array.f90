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
  ! CHECK: %[[a5:.*]] = fir.convert %arg10 : {{.*}} -> !fir.ref<f32>
  ! CHECK: fir.load %arg5 :
  ! CHECK: %[[x5:.*]] = subi %{{.*}}, %{{.*}} :
  ! CHECK: fir.coordinate_of %[[a5]], %[[x5]] :
  ! CHECK-LABEL: EndIoStatement
  print *, a5(kk)
  ! CHECK-LABEL: BeginExternalListOutput
  ! CHECK: %[[a6:.*]] = fir.convert %arg11 : {{.*}} -> !fir.ref<i32>
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
  ! CHECK: %[[a7:.*]] = fir.convert %arg12 : {{.*}} -> !fir.ref<f32>
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
