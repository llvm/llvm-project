! RUN: bbc -emit-fir -o - %s | FileCheck %s


! CHECK-LABEL: func @_QPcompare1(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.logical<4>>{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}) {
subroutine compare1(x, c1, c2)
  character(*) c1, c2, d1, d2
  logical x, y
  x = c1 < c2
  return

! CHECK-LABEL: func @_QPcompare2(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.logical<4>>{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}) {
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
  call dd1
  call dd2
  call dd3(6)
6 continue
end

! CHECK-LABEL: func @_QPss(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) {
subroutine ss(n1)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Eny"}
  integer n17, n2
  nx = 100
  n1 = nx + 10
  return

! CHECK-LABEL: func @_QPe1(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) {
entry e1(n2, n17)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Eny"}
  ny = 200
  n2 = ny + 20
  return

  ! CHECK-LABEL: func @_QPe2(
  ! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}, %{{.*}}: !fir.ref<i32>{{.*}}) {
entry e2(n3, n1)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Eny"}

! CHECK-LABEL: func @_QPe3(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) {
entry e3(n1)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Enx"}
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Eny"}
  n1 = 30
end

! CHECK-LABEL: func @_QPjj(
! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> i32
function jj(n1)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Ejj"}
  jj = 100
  jj = jj + n1
  return

  ! CHECK-LABEL: func @_QPrr(
  ! CHECK-SAME: %{{.*}}: !fir.ref<i32>{{.*}}) -> f32
entry rr(n2)
  ! CHECK: fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Ejj"}
  rr = 200.0
  rr = rr + n2
end

! CHECK-LABEL: func @_QPhh(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.char<1,10>>{{.*}}, %{{.*}}: index{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}) -> !fir.boxchar<1>
function hh(c1)
  character(10) c1, hh, qq
  hh = c1
  return
  ! CHECK-LABEL: func @_QPqq(
  ! CHECK-SAME: %{{.*}}: !fir.ref<!fir.char<1,10>>{{.*}}, %{{.*}}: index{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}) -> !fir.boxchar<1>
entry qq(c1)
  qq = c1
end

! CHECK-LABEL: func @_QPchar_array()
function char_array()
  character(10), c(5)
! CHECK-LABEL: func @_QPchar_array_entry(
! CHECK-SAME: %{{.*}}: !fir.boxchar<1>{{.*}}) -> f32 {
entry char_array_entry(c)
end

! CHECK-LABEL: func @_QPdd1()
subroutine dd1
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name =
  ! "_QFdd1Ekk"}
  ! CHECK: br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK: %[[ten:.*]] = arith.constant 10 : i32
  ! CHECK: fir.store %[[ten:.*]] to %[[kk]] : !fir.ref<i32>
  ! CHECK: br ^bb2
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK: %[[twenty:.*]] = arith.constant 20 : i32
  ! CHECK: fir.store %[[twenty:.*]] to %[[kk]] : !fir.ref<i32>
  ! CHECK: br ^bb3
  ! CHECK: ^bb3:  // pred: ^bb2
  ! CHECK: return
  kk = 10

  ! CHECK-LABEL: func @_QPdd2()
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name =
  ! "_QFdd1Ekk"}
  ! CHECK: br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK: %[[twenty:.*]] = arith.constant 20 : i32
  ! CHECK: fir.store %[[twenty:.*]] to %[[kk]] : !fir.ref<i32>
  ! CHECK: br ^bb2
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK: return
  entry dd2
  kk = 20
  return

  ! CHECK-LABEL: func @_QPdd3
  ! CHECK: %[[dd3:[0-9]*]] = fir.alloca index {bindc_name = "dd3"}
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name =
  ! "_QFdd1Ekk"}
  ! CHECK: %[[zero:.*]] = arith.constant 0 : index
  ! CHECK: fir.store %[[zero:.*]] to %[[dd3]] : !fir.ref<index>
  ! CHECK: br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK: %[[thirty:.*]] = arith.constant 30 : i32
  ! CHECK: fir.store %[[thirty:.*]] to %[[kk:[0-9]*]] : !fir.ref<i32>
  ! CHECK: br ^bb2
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK: %[[altret:[0-9]*]] = fir.load %[[dd3]] : !fir.ref<index>
  ! CHECK: return %[[altret:[0-9]*]] : index
  entry dd3(*)
  kk = 30
end
