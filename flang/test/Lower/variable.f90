! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPs() {
subroutine s
  ! CHECK-DAG: fir.alloca !fir.heap<i32> {name = "ally"}
  integer, allocatable :: ally
  ! CHECK-DAG: fir.alloca !fir.ptr<i32> {name = "pointy"} 
  integer, pointer :: pointy
  ! CHECK-DAG: fir.alloca i32 {name = "bullseye", target}
  integer, target :: bullseye
  ! CHECK: return
end subroutine s
