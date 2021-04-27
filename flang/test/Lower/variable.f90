! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPs() {
subroutine s
  ! CHECK-DAG: fir.alloca !fir.box<!fir.heap<i32>> {{{.*}}uniq_name = "{{.*}}Eally"}
  integer, allocatable :: ally
  ! CHECK-DAG: fir.alloca !fir.box<!fir.ptr<i32>> {{{.*}}uniq_name = "{{.*}}Epointy"} 
  integer, pointer :: pointy
  ! CHECK-DAG: fir.alloca i32 {{{.*}}fir.target{{.*}}uniq_name = "{{.*}}Ebullseye"}
  integer, target :: bullseye
  ! CHECK: return
end subroutine s
