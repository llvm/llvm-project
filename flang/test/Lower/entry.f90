! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QPcompare1(%arg0: !fir.ref<!fir.logical<4>>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>)
subroutine compare1(x, c1, c2)
  character(*) c1, c2, d1, d2
  logical x, y
  x = c1 < c2
  return

! CHECK-LABEL: func @_QPcompare2(%arg0: !fir.ref<!fir.logical<4>>, %arg1: !fir.boxchar<1>, %arg2: !fir.boxchar<1>)
entry compare2(y, d2, d1)
  y = d1 < d2
end

program entries
  character(10) hh, qq, m
  character(len=4) s1, s2
  integer mm
  logical r
  s1 = 'a111'
  s2 = 'a222'
  call compare1(r, s1, s2); print*, r
  call compare2(r, s1, s2); print*, r
  call ss(mm);     print*, mm
  call e1(mm, 17); print*, mm
  call e2(17, mm); print*, mm
  call e3(mm);     print*, mm
  print*, jj(11)
  print*, rr(22)
  m = 'abcd efgh'
  print*, hh(m)
  print*, qq(m)
end

! CHECK-LABEL: func @_QPss(%arg0: !fir.ref<i32>)
subroutine ss(n1)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {name = "{{.*}}Eny"}
  integer n17, n2
  nx = 100
  n1 = nx + 10
  return

! CHECK-LABEL: func @_QPe1(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>)
entry e1(n2, n17)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {name = "{{.*}}Eny"}
  ny = 200
  n2 = ny + 20
  return

! CHECK-LABEL: func @_QPe2(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>)
entry e2(n3, n1)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {name = "{{.*}}Eny"}

! CHECK-LABEL: func @_QPe3(%arg0: !fir.ref<i32>)
entry e3(n1)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {name = "{{.*}}Eny"}
  n1 = 30
end

! CHECK-LABEL: func @_QPjj(%arg0: !fir.ref<i32>) -> i32
function jj(n1)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Ejj"}
  jj = 100
  jj = jj + n1
  return

! CHECK-LABEL: func @_QPrr(%arg0: !fir.ref<i32>) -> f32
entry rr(n2)
  ! CHECK: fir.alloca i32 {name = "{{.*}}Ejj"}
  rr = 200.0
  rr = rr + n2
end

! CHECK-LABEL: func @_QPhh(%arg0: !fir.ref<!fir.char<1,?>>, %arg1: index, %arg2: !fir.boxchar<1>) -> !fir.boxchar<1>
function hh(c1)
  character(10) c1, hh, qq
  hh = c1
  return
! CHECK-LABEL: func @_QPqq(%arg0: !fir.ref<!fir.char<1,?>>, %arg1: index, %arg2: !fir.boxchar<1>) -> !fir.boxchar<1>
entry qq(c1)
  qq = c1
end
