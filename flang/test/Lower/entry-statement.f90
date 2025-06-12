! RUN: bbc -emit-hlfir -o - %s | FileCheck %s


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
  character c(3)
  character(10) hh, qq, m
  character(len=4) s1, s2
  integer mm, x(3), y(5)
  logical r
  complex xx(3)
  character(5), external :: f1, f2, f3

  interface
    subroutine ashapec(asc)
      character asc(:)
    end subroutine
    subroutine ashapei(asi)
      integer asi(:)
    end subroutine
    subroutine ashapex(asx)
      complex asx(:)
    end subroutine
  end interface

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
  x = 5
  y = 7
  call level3a(x, y, 3)
  call level3b(x, y, 3)
  call ashapec(c); print*, c
  call ashapei(x); print*, x
  call ashapex(xx); print*, xx
  print *, f1(1)
  print *, f2(2)
  print *, f3()
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

subroutine dd1
! CHECK-LABEL:   func.func @_QPdd1() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdd1Ekk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i32
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           %[[VAL_3:.*]] = arith.constant 20 : i32
! CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:           cf.br ^bb3
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }
  kk = 10

  entry dd2
! CHECK-LABEL:   func.func @_QPdd2() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdd1Ekk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_2:.*]] = arith.constant 20 : i32
! CHECK:           hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }
  kk = 20
  return

  entry dd3(*)
! CHECK-LABEL:   func.func @_QPdd3() -> index {
! CHECK:           %[[VAL_0:.*]] = fir.alloca index {bindc_name = "dd3"}
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFdd1Ekk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<index>
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_4:.*]] = arith.constant 30 : i32
! CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<index>
! CHECK:           return %[[VAL_5]] : index
! CHECK:         }
  kk = 30
end

subroutine ashapec(asc)
  character asc(:)
  integer asi(:)
  complex asx(:)
  asc = '?'
  return
! CHECK-LABEL:   func.func @_QPashapec(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1>>> {fir.bindc_name = "asc"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_4]] dummy_scope %[[VAL_3]] {uniq_name = "_QFashapecEasc"} : (!fir.box<!fir.array<?x!fir.char<1>>>, index, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.char<1>>>, !fir.box<!fir.array<?x!fir.char<1>>>)
! CHECK:           %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_9:.*]] = fir.embox %[[VAL_6]](%[[VAL_8]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_11]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_13:.*]] = fir.box_addr %[[VAL_10]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_14:.*]] = fir.shape_shift %[[VAL_12]]#0, %[[VAL_12]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) {uniq_name = "_QFashapecEasi"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_17]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_16]](%[[VAL_18]]) : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_20]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>) -> !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_24:.*]] = fir.shape_shift %[[VAL_22]]#0, %[[VAL_22]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_23]](%[[VAL_24]]) {uniq_name = "_QFashapecEasx"} : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.heap<!fir.array<?xcomplex<f32>>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_26:.*]] = fir.address_of(@_QQclX3F) : !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_26]] typeparams %[[VAL_27]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX3F"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
! CHECK:           hlfir.assign %[[VAL_28]]#0 to %[[VAL_5]]#0 : !fir.ref<!fir.char<1>>, !fir.box<!fir.array<?x!fir.char<1>>>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }

entry ashapei(asi)
  asi = 3
  return
! CHECK-LABEL:   func.func @_QPashapei(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "asi"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_11:.*]] = fir.box_elesize %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>) -> index
! CHECK:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>) -> !fir.heap<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_13:.*]] = fir.shape_shift %[[VAL_10]]#0, %[[VAL_10]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_13]]) typeparams %[[VAL_11]] {uniq_name = "_QFashapecEasc"} : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shapeshift<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1>>>, !fir.heap<!fir.array<?x!fir.char<1>>>)
! CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFashapecEasi"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_17]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_16]](%[[VAL_18]]) : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_20]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_23:.*]] = fir.box_addr %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?xcomplex<f32>>>>) -> !fir.heap<!fir.array<?xcomplex<f32>>>
! CHECK:           %[[VAL_24:.*]] = fir.shape_shift %[[VAL_22]]#0, %[[VAL_22]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_23]](%[[VAL_24]]) {uniq_name = "_QFashapecEasx"} : (!fir.heap<!fir.array<?xcomplex<f32>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.heap<!fir.array<?xcomplex<f32>>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           %[[VAL_26:.*]] = arith.constant 3 : i32
! CHECK:           hlfir.assign %[[VAL_26]] to %[[VAL_15]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }
entry ashapex(asx)
  asx = (2.0,-2.0)
end
! CHECK-LABEL:   func.func @_QPashapex(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.box<!fir.array<?xcomplex<f32>>> {fir.bindc_name = "asx"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
! CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
! CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_10:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_11:.*]] = fir.box_elesize %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>) -> index
! CHECK:           %[[VAL_12:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>) -> !fir.heap<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_13:.*]] = fir.shape_shift %[[VAL_10]]#0, %[[VAL_10]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_12]](%[[VAL_13]]) typeparams %[[VAL_11]] {uniq_name = "_QFashapecEasc"} : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shapeshift<1>, index) -> (!fir.box<!fir.array<?x!fir.char<1>>>, !fir.heap<!fir.array<?x!fir.char<1>>>)
! CHECK:           %[[VAL_15:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_15]](%[[VAL_17]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_19:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_19]], %[[VAL_20]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_22:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[VAL_23:.*]] = fir.shape_shift %[[VAL_21]]#0, %[[VAL_21]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_24:.*]]:2 = hlfir.declare %[[VAL_22]](%[[VAL_23]]) {uniq_name = "_QFashapecEasi"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           %[[VAL_25:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_3]] {uniq_name = "_QFashapecEasx"} : (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.dscope) -> (!fir.box<!fir.array<?xcomplex<f32>>>, !fir.box<!fir.array<?xcomplex<f32>>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           hlfir.assign %{{.*}} to %[[VAL_25]]#0 : complex<f32>, !fir.box<!fir.array<?xcomplex<f32>>>
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: func @_QPlevel3a(
subroutine level3a(a, b, m)
  ! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.alloca i32 {bindc_name = "n", uniq_name = "_QFlevel3aEn"}
  integer :: a(m), b(a(m)), m
  integer :: x(n), y(x(n)), n
1 print*, m
  print*, a
  print*, b
  if (m == 3) return
! CHECK-LABEL: func @_QPlevel3b(
entry level3b(x, y, n)
  ! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.alloca i32 {bindc_name = "m", uniq_name = "_QFlevel3aEm"}
  print*, n
  print*, x
  print*, y
  if (n /= 3) goto 1
end

function f1(n1) result(res1)
  character(5) res1, f2, f3
  res1 = 'a a a'
  if (n1 == 1) return
! CHECK-LABEL:   func.func @_QPf1(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.char<1,5>>,
! CHECK-SAME:                     %[[VAL_1:.*]]: index,
! CHECK-SAME:                     %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n1"}) -> !fir.boxchar<1> {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {uniq_name = "_QFf1En1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_5:.*]] = fir.alloca i32 {bindc_name = "n2", uniq_name = "_QFf1En2"}
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] {uniq_name = "_QFf1En2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] {uniq_name = "_QFf1Eres1"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_9]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef2"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_11]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef3"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_13:.*]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_14]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_10]]#0, %[[VAL_9]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_15]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_17]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_19:.*]] = fir.emboxchar %[[VAL_12]]#0, %[[VAL_11]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_18]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           hlfir.assign %{{.*}}#0 to %[[VAL_8]]#0 : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_25:.*]] = arith.cmpi eq, %[[VAL_23]], %[[VAL_24]] : i32
! CHECK:           cf.cond_br %[[VAL_25]], ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           cf.br ^bb6
! CHECK:         ^bb3:
! CHECK:           fir.call @_QFf1Ps2(
! CHECK:           cf.cond_br %{{.*}}, ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           cf.br ^bb6
! CHECK:         ^bb5:
! CHECK:           fir.call @_QFf1Ps3(
! CHECK:           cf.br ^bb6
! CHECK:         ^bb6:
! CHECK:           %[[VAL_32:.*]] = fir.emboxchar %[[VAL_8]]#0, %[[VAL_7]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           return %[[VAL_32]] : !fir.boxchar<1>
! CHECK:         }

entry f2(n2)
! CHECK-LABEL:   func.func @_QPf2(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.char<1,5>>,
! CHECK-SAME:                     %[[VAL_1:.*]]: index,
! CHECK-SAME:                     %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n2"}) -> !fir.boxchar<1> {
! CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n1", uniq_name = "_QFf1En1"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFf1En1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_3]] {uniq_name = "_QFf1En2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_7:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_7]] {uniq_name = "_QFf1Eres1"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_9:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_9]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef2"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_11:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_11]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef3"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_13:.*]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
! CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_14]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_16:.*]] = fir.emboxchar %[[VAL_10]]#0, %[[VAL_9]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_15]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_17]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_19:.*]] = fir.emboxchar %[[VAL_12]]#0, %[[VAL_11]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_19]] to %[[VAL_18]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           fir.call @_QFf1Ps2(%
! CHECK:           cf.cond_br %{{.*}}, ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           cf.br ^bb4
! CHECK:         ^bb3:
! CHECK:           fir.call @_QFf1Ps3(
! CHECK:           cf.br ^bb4
! CHECK:         ^bb4:
! CHECK:           %[[VAL_26:.*]] = fir.emboxchar %[[VAL_10]]#0, %[[VAL_9]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           return %[[VAL_26]] : !fir.boxchar<1>
! CHECK:         }
  call s2
  if (n2 == 2) return

entry f3
! CHECK-LABEL:   func.func @_QPf3(
! CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.char<1,5>>,
! CHECK-SAME:                     %[[VAL_1:.*]]: index) -> !fir.boxchar<1> {
! CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "n1", uniq_name = "_QFf1En1"}
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFf1En1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "n2", uniq_name = "_QFf1En2"}
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFf1En2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_6]] {uniq_name = "_QFf1Eres1"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_8:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_8]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef2"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_10:.*]] = arith.constant 5 : index
! CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_10]] {fortran_attrs = #fir.var_attrs<internal_assoc>, uniq_name = "_QFf1Ef3"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:           %[[VAL_12:.*]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
! CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_13]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_15:.*]] = fir.emboxchar %[[VAL_9]]#0, %[[VAL_8]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_14]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_17:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_16]] : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
! CHECK:           %[[VAL_18:.*]] = fir.emboxchar %[[VAL_11]]#0, %[[VAL_10]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           fir.store %[[VAL_18]] to %[[VAL_17]] : !fir.ref<!fir.boxchar<1>>
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           fir.call @_QFf1Ps3(
! CHECK:           cf.br ^bb2
! CHECK:         ^bb2:
! CHECK:           %[[VAL_22:.*]] = fir.emboxchar %[[VAL_11]]#0, %[[VAL_10]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:           return %[[VAL_22]] : !fir.boxchar<1>
! CHECK:         }
  f3 = "C C C"
  call s3
contains
  subroutine s2
    f2 = 'b b b'
  end

  subroutine s3
    f3 = 'c c c'
  end
end

subroutine assumed_size()
  real :: x(*)
! CHECK-LABEL:   func.func @_QPassumed_size() {
! CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[VAL_1:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_8:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           %[[VAL_9:.*]] = fir.shape_shift %[[VAL_7]]#0, %[[VAL_7]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {uniq_name = "_QFassumed_sizeEx"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           return
! CHECK:         }

  entry entry_with_assumed_size(x)
end subroutine
! CHECK-LABEL:   func.func @_QPentry_with_assumed_size(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) dummy_scope %[[VAL_1]] {uniq_name = "_QFassumed_sizeEx"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           return
! CHECK:         }
