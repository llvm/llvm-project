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

! CHECK-LABEL: func @_QPdd1()
subroutine dd1
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
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
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
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
  ! CHECK: %[[kk:[0-9]*]] = fir.alloca i32 {bindc_name = "kk", uniq_name = "_QFdd1Ekk"}
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

! CHECK-LABEL: func @_QPashapec(
subroutine ashapec(asc)
  ! CHECK: %[[asx:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>
  ! CHECK: %[[asi:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: %[[zeroi:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[shapei:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxi:[0-9]*]] = fir.embox %[[zeroi]](%[[shapei]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.store %[[boxi]] to %[[asi]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  ! CHECK: %[[zerox:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.complex<4>>>
  ! CHECK: %[[shapex:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxx:[0-9].*]] = fir.embox %[[zerox]](%[[shapex]]) : (!fir.heap<!fir.array<?x!fir.complex<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>
  ! CHECK: fir.store %[[boxx]] to %[[asx]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>>
  character asc(:)
  integer asi(:)
  complex asx(:)
  asc = '?'
  return
! CHECK-LABEL: func @_QPashapei(
entry ashapei(asi)
  ! CHECK: %[[asx:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>
  ! CHECK: %[[asc:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
  ! CHECK: %[[zeroc:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[shapec:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxc:[0-9]*]] = fir.embox %[[zeroc]](%[[shapec]]) : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
  ! CHECK: fir.store %[[boxc]] to %[[asc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
  ! CHECK: %[[zerox:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.complex<4>>>
  ! CHECK: %[[shapex:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxx:[0-9].*]] = fir.embox %[[zerox]](%[[shapex]]) : (!fir.heap<!fir.array<?x!fir.complex<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>
  ! CHECK: fir.store %[[boxx]] to %[[asx]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.complex<4>>>>>
  asi = 3
  return
! CHECK-LABEL: func @_QPashapex(
entry ashapex(asx)
  ! CHECK: %[[asi:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: %[[asc:[0-9]*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
  ! CHECK: %[[zeroc:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1>>>
  ! CHECK: %[[shapec:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxc:[0-9]*]] = fir.embox %[[zeroc]](%[[shapec]]) : (!fir.heap<!fir.array<?x!fir.char<1>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>
  ! CHECK: fir.store %[[boxc]] to %[[asc]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1>>>>>
  ! CHECK: %[[zeroi:[0-9]*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  ! CHECK: %[[shapei:[0-9]*]] = fir.shape %c0{{.*}} : (index) -> !fir.shape<1>
  ! CHECK: %[[boxi:[0-9].*]] = fir.embox %[[zeroi]](%[[shapei]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK: fir.store %[[boxi]] to %[[asi]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  asx = (2.0,-2.0)
end

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

! CHECK-LABEL: @_QPf1
function f1(n1) result(res1)
  ! CHECK:   %[[V_0:[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca i32 {bindc_name = "n2", uniq_name = "_QFf1En2"}
  ! CHECK:   %[[V_2:[0-9]+]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
  ! CHECK:   %[[V_3:[0-9]+]] = fir.coordinate_of %[[V_2]], %c0{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_4]] to %[[V_3]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.coordinate_of %[[V_2]], %c1{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_6]] to %[[V_5]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK:   %[[V_7:[0-9]+]] = fir.address_of(@_QQcl.6120612061) : !fir.ref<!fir.char<1,5>>
  ! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi slt, %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_9:[0-9]+]] = arith.select %[[V_8]], %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_10:[0-9]+]] = fir.convert %[[V_9]] : (index) -> i64
  ! CHECK:   %[[V_11:[0-9]+]] = arith.muli %c1{{.*}}_i64, %[[V_10]] : i64
  ! CHECK:   %[[V_12:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:   %[[V_13:[0-9]+]] = fir.convert %[[V_7]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_12]], %[[V_13]], %[[V_11]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:   %[[V_14:[0-9]+]] = arith.subi %c5{{.*}}, %c1{{.*}} : index
  ! CHECK:   %[[V_15:[0-9]+]] = fir.undefined !fir.char<1>
  ! CHECK:   %[[V_16:[0-9]+]] = fir.insert_value %[[V_15]], %c32{{.*}}_i8, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:   fir.do_loop %arg3 = %[[V_9]] to %[[V_14]] step %c1{{.*}} {
  ! CHECK:     %[[V_32:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:     %[[V_33:[0-9]+]] = fir.coordinate_of %[[V_32]], %arg3 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:     fir.store %[[V_16]] to %[[V_33]] : !fir.ref<!fir.char<1>>
  ! CHECK:   }
  ! CHECK:   %[[V_17:[0-9]+]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK:   %[[V_18:[0-9]+]] = arith.cmpi eq, %[[V_17]], %c1{{.*}}_i32_4 : i32
  ! CHECK:   cond_br %[[V_18]], ^bb2, ^bb3
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK:   br ^bb6
  ! CHECK: ^bb3:  // pred: ^bb1
  ! CHECK:   fir.call @_QFf1Ps2(%[[V_2]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>) -> ()
  ! CHECK:   %[[V_19:[0-9]+]] = fir.load %[[V_1]] : !fir.ref<i32>
  ! CHECK:   %[[V_20:[0-9]+]] = arith.cmpi eq, %[[V_19]], %c2{{.*}}_i32 : i32
  ! CHECK:   cond_br %[[V_20]], ^bb4, ^bb5
  ! CHECK: ^bb4:  // pred: ^bb3
  ! CHECK:   cf.br ^bb6
  ! CHECK: ^bb5:  // pred: ^bb3
  ! CHECK:   %[[V_21:[0-9]+]] = fir.address_of(@_QQcl.4320432043) : !fir.ref<!fir.char<1,5>>
  ! CHECK:   %[[V_22:[0-9]+]] = arith.cmpi slt, %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_23:[0-9]+]] = arith.select %[[V_22]], %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_24:[0-9]+]] = fir.convert %[[V_23]] : (index) -> i64
  ! CHECK:   %[[V_25:[0-9]+]] = arith.muli %c1{{.*}}_i64_6, %[[V_24]] : i64
  ! CHECK:   %[[V_26:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:   %[[V_27:[0-9]+]] = fir.convert %[[V_21]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_26]], %[[V_27]], %[[V_25]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:   %[[V_28:[0-9]+]] = arith.subi %c5{{.*}}, %c1{{.*}} : index
  ! CHECK:   %[[V_29:[0-9]+]] = fir.undefined !fir.char<1>
  ! CHECK:   %[[V_30:[0-9]+]] = fir.insert_value %[[V_29]], %c32{{.*}}_i8_9, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:   fir.do_loop %arg3 = %[[V_23]] to %[[V_28]] step %c1{{.*}} {
  ! CHECK:     %[[V_32]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:     %[[V_33]] = fir.coordinate_of %[[V_32]], %arg3 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:     fir.store %[[V_30]] to %[[V_33]] : !fir.ref<!fir.char<1>>
  ! CHECK:   }
  ! CHECK:   fir.call @_QFf1Ps3(%[[V_2]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>) -> ()
  ! CHECK:   br ^bb6
  ! CHECK: ^bb6:  //  3 preds: ^bb2, ^bb4, ^bb5
  ! CHECK:   %[[V_31:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   return %[[V_31]] : !fir.boxchar<1>
  ! CHECK: }
  character(5) res1, f2, f3
  res1 = 'a a a'
  if (n1 == 1) return

! CHECK-LABEL: @_QPf2
entry f2(n2)
  ! CHECK:   %[[V_0:[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca i32 {bindc_name = "n1", uniq_name = "_QFf1En1"}
  ! CHECK:   %[[V_2:[0-9]+]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
  ! CHECK:   %[[V_3:[0-9]+]] = fir.coordinate_of %[[V_2]], %c0{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_4]] to %[[V_3]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.coordinate_of %[[V_2]], %c1{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_6]] to %[[V_5]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK:   fir.call @_QFf1Ps2(%[[V_2]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>) -> ()
  ! CHECK:   %[[V_7:[0-9]+]] = fir.load %arg2 : !fir.ref<i32>
  ! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi eq, %[[V_7]], %c2{{.*}}_i32 : i32
  ! CHECK:   cond_br %[[V_8]], ^bb2, ^bb3
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK:   br ^bb4
  ! CHECK: ^bb3:  // pred: ^bb1
  ! CHECK:   %[[V_9:[0-9]+]] = fir.address_of(@_QQcl.4320432043) : !fir.ref<!fir.char<1,5>>
  ! CHECK:   %[[V_10:[0-9]+]] = arith.cmpi slt, %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_11:[0-9]+]] = arith.select %[[V_10]], %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_12:[0-9]+]] = fir.convert %[[V_11]] : (index) -> i64
  ! CHECK:   %[[V_13:[0-9]+]] = arith.muli %c1{{.*}}_i64, %[[V_12]] : i64
  ! CHECK:   %[[V_14:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:   %[[V_15:[0-9]+]] = fir.convert %[[V_9]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_14]], %[[V_15]], %[[V_13]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:   %[[V_16:[0-9]+]] = arith.subi %c5{{.*}}, %c1{{.*}} : index
  ! CHECK:   %[[V_17:[0-9]+]] = fir.undefined !fir.char<1>
  ! CHECK:   %[[V_18:[0-9]+]] = fir.insert_value %[[V_17]], %c32{{.*}}_i8, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:   fir.do_loop %arg3 = %[[V_11]] to %[[V_16]] step %c1{{.*}} {
  ! CHECK:     %[[V_20:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:     %[[V_21:[0-9]+]] = fir.coordinate_of %[[V_20]], %arg3 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:     fir.store %[[V_18]] to %[[V_21]] : !fir.ref<!fir.char<1>>
  ! CHECK:   }
  ! CHECK:   fir.call @_QFf1Ps3(%[[V_2]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>) -> ()
  ! CHECK:   br ^bb4
  ! CHECK: ^bb4:  // 2 preds: ^bb2, ^bb3
  ! CHECK:   %[[V_19:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   return %[[V_19]] : !fir.boxchar<1>
  ! CHECK: }
  call s2
  if (n2 == 2) return

! CHECK-LABEL: @_QPf3
entry f3
  ! CHECK:   %[[V_0:[0-9]+]] = fir.convert %arg0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca i32 {bindc_name = "n1", uniq_name = "_QFf1En1"}
  ! CHECK:   %[[V_2:[0-9]+]] = fir.alloca i32 {bindc_name = "n2", uniq_name = "_QFf1En2"}
  ! CHECK:   %[[V_3:[0-9]+]] = fir.alloca tuple<!fir.boxchar<1>, !fir.boxchar<1>>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.coordinate_of %[[V_3]], %c0{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_5]] to %[[V_4]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.coordinate_of %[[V_3]], %c1{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK:   %[[V_7:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   fir.store %[[V_7]] to %[[V_6]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK:   br ^bb1
  ! CHECK: ^bb1:  // pred: ^bb0
  ! CHECK:   %[[V_8:[0-9]+]] = fir.address_of(@_QQcl.4320432043) : !fir.ref<!fir.char<1,5>>
  ! CHECK:   %[[V_9:[0-9]+]] = arith.cmpi slt, %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_10:[0-9]+]] = arith.select %[[V_9]], %c5{{.*}}, %c5{{.*}} : index
  ! CHECK:   %[[V_11:[0-9]+]] = fir.convert %[[V_10]] : (index) -> i64
  ! CHECK:   %[[V_12:[0-9]+]] = arith.muli %c1{{.*}}_i64, %[[V_11]] : i64
  ! CHECK:   %[[V_13:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:   %[[V_14:[0-9]+]] = fir.convert %[[V_8]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_13]], %[[V_14]], %[[V_12]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:   %[[V_15:[0-9]+]] = arith.subi %c5{{.*}}, %c1{{.*}} : index
  ! CHECK:   %[[V_16:[0-9]+]] = fir.undefined !fir.char<1>
  ! CHECK:   %[[V_17:[0-9]+]] = fir.insert_value %[[V_16]], %c32{{.*}}_i8, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:   fir.do_loop %arg2 = %[[V_10]] to %[[V_15]] step %c1{{.*}} {
  ! CHECK:     %[[V_19:[0-9]+]] = fir.convert %[[V_0]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:     %[[V_20:[0-9]+]] = fir.coordinate_of %[[V_19]], %arg2 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:     fir.store %[[V_17]] to %[[V_20]] : !fir.ref<!fir.char<1>>
  ! CHECK:   }
  ! CHECK:   fir.call @_QFf1Ps3(%[[V_3]]) {{.*}}: (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>) -> ()
  ! CHECK:   br ^bb2
  ! CHECK: ^bb2:  // pred: ^bb1
  ! CHECK:   %[[V_18:[0-9]+]] = fir.emboxchar %[[V_0]], %c5{{.*}} : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:   return %[[V_18]] : !fir.boxchar<1>
  ! CHECK: }
  f3 = "C C C"
  call s3
contains
  ! CHECK-LABEL: @_QFf1Ps2
  subroutine s2
    ! CHECK:   %[[V_0:[0-9]+]] = fir.coordinate_of %arg0, %c0{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
    ! CHECK:   %[[V_1:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<!fir.boxchar<1>>
    ! CHECK:   %[[V_2:[0-9]+]]:2 = fir.unboxchar %[[V_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK:   %[[V_3:[0-9]+]] = fir.address_of(@_QQcl.6220622062) : !fir.ref<!fir.char<1,5>>
    ! CHECK:   %[[V_4:[0-9]+]] = arith.cmpi slt, %[[V_2]]#1, %c5{{.*}} : index
    ! CHECK:   %[[V_5:[0-9]+]] = arith.select %[[V_4]], %[[V_2]]#1, %c5{{.*}} : index
    ! CHECK:   %[[V_6:[0-9]+]] = fir.convert %[[V_5]] : (index) -> i64
    ! CHECK:   %[[V_7:[0-9]+]] = arith.muli %c1{{.*}}_i64, %[[V_6]] : i64
    ! CHECK:   %[[V_8:[0-9]+]] = fir.convert %[[V_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK:   %[[V_9:[0-9]+]] = fir.convert %[[V_3]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
    ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_8]], %[[V_9]], %[[V_7]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
    ! CHECK:   %[[V_10:[0-9]+]] = arith.subi %[[V_2]]#1, %c1{{.*}} : index
    ! CHECK:   %[[V_11:[0-9]+]] = fir.undefined !fir.char<1>
    ! CHECK:   %[[V_12:[0-9]+]] = fir.insert_value %[[V_11]], %c32{{.*}}_i8, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
    ! CHECK:   fir.do_loop %arg1 = %[[V_5]] to %[[V_10]] step %c1{{.*}} {
    ! CHECK:     %[[V_13:[0-9]+]] = fir.convert %[[V_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK:     %[[V_14:[0-9]+]] = fir.coordinate_of %[[V_13]], %arg1 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    ! CHECK:     fir.store %[[V_12]] to %[[V_14]] : !fir.ref<!fir.char<1>>
    ! CHECK:   }
    ! CHECK:   return
    ! CHECK: }
    f2 = 'b b b'
  end

  ! CHECK-LABEL: @_QFf1Ps3
  subroutine s3
    ! CHECK:   %[[V_0:[0-9]+]] = fir.coordinate_of %arg0, %c1{{.*}}_i32 : (!fir.ref<tuple<!fir.boxchar<1>, !fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
    ! CHECK:   %[[V_1:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<!fir.boxchar<1>>
    ! CHECK:   %[[V_2:[0-9]+]]:2 = fir.unboxchar %[[V_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    ! CHECK:   %[[V_3:[0-9]+]] = fir.address_of(@_QQcl.6320632063) : !fir.ref<!fir.char<1,5>>
    ! CHECK:   %[[V_4:[0-9]+]] = arith.cmpi slt, %[[V_2]]#1, %c5{{.*}} : index
    ! CHECK:   %[[V_5:[0-9]+]] = arith.select %[[V_4]], %[[V_2]]#1, %c5{{.*}} : index
    ! CHECK:   %[[V_6:[0-9]+]] = fir.convert %[[V_5]] : (index) -> i64
    ! CHECK:   %[[V_7:[0-9]+]] = arith.muli %c1{{.*}}_i64, %[[V_6]] : i64
    ! CHECK:   %[[V_8:[0-9]+]] = fir.convert %[[V_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
    ! CHECK:   %[[V_9:[0-9]+]] = fir.convert %[[V_3]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
    ! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[V_8]], %[[V_9]], %[[V_7]], %false{{.*}}) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
    ! CHECK:   %[[V_10:[0-9]+]] = arith.subi %[[V_2]]#1, %c1{{.*}} : index
    ! CHECK:   %[[V_11:[0-9]+]] = fir.undefined !fir.char<1>
    ! CHECK:   %[[V_12:[0-9]+]] = fir.insert_value %[[V_11]], %c32{{.*}}_i8, [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
    ! CHECK:   fir.do_loop %arg1 = %[[V_5]] to %[[V_10]] step %c1{{.*}} {
    ! CHECK:     %[[V_13:[0-9]+]] = fir.convert %[[V_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
    ! CHECK:     %[[V_14:[0-9]+]] = fir.coordinate_of %[[V_13]], %arg1 : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
    ! CHECK:     fir.store %[[V_12]] to %[[V_14]] : !fir.ref<!fir.char<1>>
    ! CHECK:   }
    ! CHECK:   return
    ! CHECK: }
    f3 = 'c c c'
  end
end

! CHECK-LABEL: func @_QPassumed_size() {
subroutine assumed_size()
  real :: x(*)
! CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  %[[VAL_1:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:  fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  return
! CHECK:  }

! CHECK-LABEL: func @_QPentry_with_assumed_size(
  entry entry_with_assumed_size(x)
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
! CHECK:  br ^bb1
! CHECK:  ^bb1:
! CHECK:  return
! CHECK:  }
end subroutine
