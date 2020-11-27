! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPs() {
subroutine s
  ! CHECK-DAG: fir.alloca !fir.box<!fir.heap<i32>> {name = "{{.*}}Eally"}
  integer, allocatable :: ally
  ! CHECK-DAG: fir.alloca !fir.box<!fir.ptr<i32>> {name = "{{.*}}Epointy"} 
  integer, pointer :: pointy
  ! CHECK-DAG: fir.alloca i32 {name = "{{.*}}Ebullseye", target}
  integer, target :: bullseye
  ! CHECK: return
end subroutine s
