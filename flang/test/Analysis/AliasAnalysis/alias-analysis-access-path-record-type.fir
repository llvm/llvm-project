// Tests for access paths in fir::AliasAnalysis through views that change the
// type of the addressed object: CLASS views of objects of extended types,
// SELECT TYPE, storage association, sequence association, and component
// sections of arrays, and through dereferences of allocatable and pointer
// components. The parent type of an extended type is given by its
// fir.type_info.

// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading  \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' \
// RUN:   2>&1 | FileCheck -match-full-lines %s

// -----

// A CLASS(t1) view of an object of extended type t2, as created when a
// type-bound procedure with a CLASS(t1) passed-object dummy is inlined.
// view_a is obj%t1%a(1) reached through the view.

// CHECK-LABEL: Testing : "_QPparent_view"
// CHECK-DAG: direct_a#0 <-> direct_b#0: NoAlias
// CHECK-DAG: direct_a#0 <-> view_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> view_a#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
func.func @_QPparent_view() {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c4 : (index) -> !fir.shape<1>
  %0 = fir.alloca !t2 {bindc_name = "obj", uniq_name = "_QFparent_viewEobj"}
  %obj = fir.declare %0 uniq_name("_QFparent_viewEobj") : (!fir.ref<!t2>) -> !fir.ref<!t2>
  %parent = fir.coordinate_of %obj, t1 : (!fir.ref<!t2>) -> !fir.ref<!t1>
  %pa = fir.coordinate_of %parent, a : (!fir.ref<!t1>) -> !fir.ref<!fir.array<4xf32>>
  %direct_a = fir.array_coor %pa(%sh) %c1 {test.ptr = "direct_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %direct_b = fir.coordinate_of %obj, b {test.ptr = "direct_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %box = fir.embox %obj : (!fir.ref<!t2>) -> !fir.box<!t2>
  %view = fir.convert %box : (!fir.box<!t2>) -> !fir.class<!t1>
  %ds = fir.dummy_scope : !fir.dscope
  %this = fir.declare %view dummy_scope %ds arg 1 uniq_name("_QMmFgetEthis") : (!fir.class<!t1>, !fir.dscope) -> !fir.class<!t1>
  %va = fir.coordinate_of %this, a : (!fir.class<!t1>) -> !fir.ref<!fir.array<4xf32>>
  %view_a = fir.array_coor %va(%sh) %c1 {test.ptr = "view_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return
}

// -----

// CLASS(t1) array views of an array of objects of extended type t2.

// CHECK-LABEL: Testing : "_QParray_view"
// CHECK-DAG: direct_a#0 <-> direct_b#0: NoAlias
// CHECK-DAG: direct_a#0 <-> view_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> view_a#0: NoAlias
// CHECK-DAG: direct_a#0 <-> rebox_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> rebox_a#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
func.func @_QParray_view(%i: index) {
  %c10 = arith.constant 10 : index
  %sh = fir.shape %c10 : (index) -> !fir.shape<1>
  %0 = fir.alloca !fir.array<10x!t2> {bindc_name = "arr", uniq_name = "_QFarray_viewEarr"}
  %arr = fir.declare %0(%sh) uniq_name("_QFarray_viewEarr") : (!fir.ref<!fir.array<10x!t2>>, !fir.shape<1>) -> !fir.ref<!fir.array<10x!t2>>
  %elt = fir.array_coor %arr(%sh) %i : (!fir.ref<!fir.array<10x!t2>>, !fir.shape<1>, index) -> !fir.ref<!t2>
  %direct_a = fir.coordinate_of %elt, t1, a {test.ptr = "direct_a"} : (!fir.ref<!t2>) -> !fir.ref<!fir.array<4xf32>>
  %direct_b = fir.coordinate_of %elt, b {test.ptr = "direct_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %box = fir.embox %arr(%sh) : (!fir.ref<!fir.array<10x!t2>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!t2>>
  %view = fir.convert %box : (!fir.box<!fir.array<10x!t2>>) -> !fir.class<!fir.array<?x!t1>>
  %velt = fir.array_coor %view %i : (!fir.class<!fir.array<?x!t1>>, index) -> !fir.ref<!t1>
  %view_a = fir.coordinate_of %velt, a {test.ptr = "view_a"} : (!fir.ref<!t1>) -> !fir.ref<!fir.array<4xf32>>
  %rebox = fir.rebox %box : (!fir.box<!fir.array<10x!t2>>) -> !fir.class<!fir.array<?x!t1>>
  %relt = fir.array_coor %rebox %i : (!fir.class<!fir.array<?x!t1>>, index) -> !fir.ref<!t1>
  %rebox_a = fir.coordinate_of %relt, a {test.ptr = "rebox_a"} : (!fir.ref<!t1>) -> !fir.ref<!fir.array<4xf32>>
  return
}

// -----

// A CLASS(t1) view of an object of type t3, which extends t2, which extends
// t1. view_a is obj%t2%t1%a.

// CHECK-LABEL: Testing : "_QPtwo_level"
// CHECK-DAG: direct_a#0 <-> view_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> view_a#0: NoAlias
// CHECK-DAG: direct_c#0 <-> view_a#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
!t3 = !fir.type<_QMmTt3{t2:!t2,c:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
fir.type_info @_QMmTt3 extends !t2 : !t3
func.func @_QPtwo_level() {
  %0 = fir.alloca !t3 {bindc_name = "obj", uniq_name = "_QFtwo_levelEobj"}
  %obj = fir.declare %0 uniq_name("_QFtwo_levelEobj") : (!fir.ref<!t3>) -> !fir.ref<!t3>
  %direct_a = fir.coordinate_of %obj, t2, t1, a {test.ptr = "direct_a"} : (!fir.ref<!t3>) -> !fir.ref<!fir.array<4xf32>>
  %direct_b = fir.coordinate_of %obj, t2, b {test.ptr = "direct_b"} : (!fir.ref<!t3>) -> !fir.ref<f32>
  %direct_c = fir.coordinate_of %obj, c {test.ptr = "direct_c"} : (!fir.ref<!t3>) -> !fir.ref<f32>
  %box = fir.embox %obj : (!fir.ref<!t3>) -> !fir.box<!t3>
  %view = fir.convert %box : (!fir.box<!t3>) -> !fir.class<!t1>
  %view_a = fir.coordinate_of %view, a {test.ptr = "view_a"} : (!fir.class<!t1>) -> !fir.ref<!fir.array<4xf32>>
  return
}

// -----

// SELECT TYPE with TYPE IS (t2) on the CLASS(t1) view of an object of type
// t2, as created when the type-bound procedure is inlined. The selector
// addresses the whole object again, so view_b is obj%b, and view_a is
// obj%t1%a.

// CHECK-LABEL: Testing : "_QPup_down"
// CHECK-DAG: direct_a#0 <-> view_b#0: NoAlias
// CHECK-DAG: direct_b#0 <-> view_b#0: MayAlias
// CHECK-DAG: direct_a#0 <-> view_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> view_a#0: NoAlias
// CHECK-DAG: view_b#0 <-> view_a#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
func.func @_QPup_down() {
  %0 = fir.alloca !t2 {bindc_name = "obj", uniq_name = "_QFup_downEobj"}
  %obj = fir.declare %0 uniq_name("_QFup_downEobj") : (!fir.ref<!t2>) -> !fir.ref<!t2>
  %direct_a = fir.coordinate_of %obj, t1, a {test.ptr = "direct_a"} : (!fir.ref<!t2>) -> !fir.ref<!fir.array<4xf32>>
  %direct_b = fir.coordinate_of %obj, b {test.ptr = "direct_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %box = fir.embox %obj : (!fir.ref<!t2>) -> !fir.box<!t2>
  %view = fir.convert %box : (!fir.box<!t2>) -> !fir.class<!t1>
  %ds = fir.dummy_scope : !fir.dscope
  %this = fir.declare %view dummy_scope %ds arg 1 uniq_name("_QMmFgetEthis") : (!fir.class<!t1>, !fir.dscope) -> !fir.class<!t1>
  %addr = fir.box_addr %this : (!fir.class<!t1>) -> !fir.ref<!t2>
  %sel = fir.declare %addr uniq_name("_QMmFgetEsel") : (!fir.ref<!t2>) -> !fir.ref<!t2>
  %view_b = fir.coordinate_of %sel, b {test.ptr = "view_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %view_a = fir.coordinate_of %sel, t1, a {test.ptr = "view_a"} : (!fir.ref<!t2>) -> !fir.ref<!fir.array<4xf32>>
  return
}

// -----

// SELECT TYPE with TYPE IS (t2) on a CLASS(t1) dummy:
//   select type (sel => this)
//   type is (t2)
//     ... sel%a ... this%a ... sel%b
// sel%a and this%a are the same component. this%a and sel%b are disjoint.

// CHECK-LABEL: Testing : "_QPsel"
// CHECK-DAG: this_a#0 <-> sel_a#0: MayAlias
// CHECK-DAG: this_a#0 <-> sel_b#0: NoAlias
// CHECK-DAG: sel_a#0 <-> sel_b#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
func.func @_QPsel(%arg0: !fir.class<!t1> {fir.bindc_name = "this"}) {
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c4 : (index) -> !fir.shape<1>
  %ds = fir.dummy_scope : !fir.dscope
  %this:2 = hlfir.declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFselEthis") : (!fir.class<!t1>, !fir.dscope) -> (!fir.class<!t1>, !fir.class<!t1>)
  %addr = fir.box_addr %this#1 : (!fir.class<!t1>) -> !fir.ref<!t2>
  %sel:2 = hlfir.declare %addr uniq_name("_QFselEsel") : (!fir.ref<!t2>) -> (!fir.ref<!t2>, !fir.ref<!t2>)
  %this_a = hlfir.designate %this#0{"a"} shape %sh {test.ptr = "this_a"} : (!fir.class<!t1>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  %parent = hlfir.designate %sel#0{"t1"} : (!fir.ref<!t2>) -> !fir.ref<!t1>
  %sel_a = hlfir.designate %parent{"a"} shape %sh {test.ptr = "sel_a"} : (!fir.ref<!t1>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  %sel_b = hlfir.designate %sel#0{"b"} {test.ptr = "sel_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  return
}

// -----

// SELECT TYPE with TYPE IS (child) on a CLASS(base) dummy, where the
// selector is accessed through the parent component:
//   select type (a => this)
//   type is (child)
//     this%v(1:3) = a%base%v(2:4)
// The two sections overlap.

// CHECK-LABEL: Testing : "_QMmPshift"
// CHECK-DAG: this_v#0 <-> a_v#0: MayAlias

!base = !fir.type<_QMmTbase{v:!fir.array<4xf32>}>
!child = !fir.type<_QMmTchild{base:!base,extra:!fir.array<4xf32>}>
fir.type_info @_QMmTbase : !base
fir.type_info @_QMmTchild extends !base : !child
func.func @_QMmPshift(%arg0: !fir.class<!base> {fir.bindc_name = "this"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QMmFshiftEthis") : (!fir.class<!base>, !fir.dscope) -> (!fir.class<!base>, !fir.class<!base>)
  %2 = fir.box_addr %1#1 : (!fir.class<!base>) -> !fir.ref<!child>
  %3:2 = hlfir.declare %2 uniq_name("_QMmFshiftEa") : (!fir.ref<!child>) -> (!fir.ref<!child>, !fir.ref<!child>)
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %5 = fir.shape %c3 : (index) -> !fir.shape<1>
  %6 = hlfir.designate %1#0{"v"} <%4> (%c1:%c3:%c1)  shape %5 {test.ptr = "this_v"} : (!fir.class<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %7 = hlfir.designate %3#0{"base"}   : (!fir.ref<!child>) -> !fir.ref<!base>
  %8 = hlfir.designate %7{"v"} <%4> (%c2:%c4:%c1)  shape %5 {test.ptr = "a_v"} : (!fir.ref<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  return
}

// -----

// The same SELECT TYPE after conversion to FIR and inlining into a caller
// (two dummy scopes).

// CHECK-LABEL: Testing : "_QMmPshift_inlined"
// CHECK-DAG: this_v#0 <-> a_v#0: MayAlias

!base = !fir.type<_QMmTbase{v:!fir.array<4xf32>}>
!child = !fir.type<_QMmTchild{base:!base,extra:!fir.array<4xf32>}>
fir.type_info @_QMmTbase : !base
fir.type_info @_QMmTchild extends !base : !child
func.func @_QMmPshift_inlined(%arg0: !fir.class<!base> {fir.bindc_name = "this"}, %i: index) {
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c4 : (index) -> !fir.shape<1>
  %caller_scope = fir.dummy_scope : !fir.dscope
  %ds = fir.dummy_scope : !fir.dscope
  %this = fir.declare %arg0 dummy_scope %ds arg 1 uniq_name("_QMmFshiftEthis") : (!fir.class<!base>, !fir.dscope) -> !fir.class<!base>
  %2 = fir.box_addr %this : (!fir.class<!base>) -> !fir.ref<!child>
  %a = fir.declare %2 uniq_name("_QMmFshiftEa") : (!fir.ref<!child>) -> !fir.ref<!child>
  %tv = fir.coordinate_of %this, v : (!fir.class<!base>) -> !fir.ref<!fir.array<4xf32>>
  %this_v = fir.array_coor %tv(%sh) %i {test.ptr = "this_v"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %av = fir.coordinate_of %a, base, v : (!fir.ref<!child>) -> !fir.ref<!fir.array<4xf32>>
  %a_v = fir.array_coor %av(%sh) %i {test.ptr = "a_v"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return
}

// -----

// CLASS(*) views of an object of type t2, and SELECT TYPE on them with
// TYPE IS (t2) (view_b is obj%b) and CLASS IS (t1) (view_a is obj%t1%a).

// CHECK-LABEL: Testing : "_QPunlimited"
// CHECK-DAG: direct_a#0 <-> view_b#0: NoAlias
// CHECK-DAG: direct_b#0 <-> view_b#0: MayAlias
// CHECK-DAG: direct_a#0 <-> view_a#0: MayAlias
// CHECK-DAG: direct_b#0 <-> view_a#0: NoAlias
// CHECK-DAG: view_b#0 <-> view_a#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
fir.type_info @_QMmTt1 : !t1
fir.type_info @_QMmTt2 extends !t1 : !t2
func.func @_QPunlimited() {
  %0 = fir.alloca !t2 {bindc_name = "obj", uniq_name = "_QFunlimitedEobj"}
  %obj = fir.declare %0 uniq_name("_QFunlimitedEobj") : (!fir.ref<!t2>) -> !fir.ref<!t2>
  %direct_a = fir.coordinate_of %obj, t1, a {test.ptr = "direct_a"} : (!fir.ref<!t2>) -> !fir.ref<!fir.array<4xf32>>
  %direct_b = fir.coordinate_of %obj, b {test.ptr = "direct_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %box = fir.embox %obj : (!fir.ref<!t2>) -> !fir.box<!t2>
  %any = fir.convert %box : (!fir.box<!t2>) -> !fir.class<none>
  %addr = fir.box_addr %any : (!fir.class<none>) -> !fir.ref<!t2>
  %view_b = fir.coordinate_of %addr, b {test.ptr = "view_b"} : (!fir.ref<!t2>) -> !fir.ref<f32>
  %cls = fir.convert %any : (!fir.class<none>) -> !fir.class<!t1>
  %view_a = fir.coordinate_of %cls, a {test.ptr = "view_a"} : (!fir.class<!t1>) -> !fir.ref<!fir.array<4xf32>>
  return
}

// -----

// EQUIVALENCE of variables of different sequence types:
//   type(s1) :: x   ! components a, b
//   type(s2) :: y   ! components c, d
//   equivalence (x, y)
// x%b and y%d are the same memory. The components of x are still disjoint.

// CHECK-LABEL: Testing : "_QPequiv"
// CHECK-DAG: x_a#0 <-> y_d#0: MayAlias
// CHECK-DAG: x_a#0 <-> x_b#0: NoAlias
// CHECK-DAG: y_d#0 <-> x_b#0: MayAlias

!s1 = !fir.type<_QFequivTs1,sequence{a:f32,b:f32}>
!s2 = !fir.type<_QFequivTs2,sequence{c:f32,d:f32}>
func.func @_QPequiv() {
  %c0 = arith.constant 0 : index
  %0 = fir.alloca !fir.array<8xi8> {uniq_name = "_QFequivEx"}
  %1 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  %2 = fir.convert %1 : (!fir.ref<i8>) -> !fir.ptr<!s1>
  %x = fir.declare %2 storage(%0[0]) uniq_name("_QFequivEx") : (!fir.ptr<!s1>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<!s1>
  %3 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<8xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ptr<!s2>
  %y = fir.declare %4 storage(%0[0]) uniq_name("_QFequivEy") : (!fir.ptr<!s2>, !fir.ref<!fir.array<8xi8>>) -> !fir.ptr<!s2>
  %x_a = fir.coordinate_of %x, a {test.ptr = "x_a"} : (!fir.ptr<!s1>) -> !fir.ref<f32>
  %y_d = fir.coordinate_of %y, d {test.ptr = "y_d"} : (!fir.ptr<!s2>) -> !fir.ref<f32>
  %x_b = fir.coordinate_of %x, b {test.ptr = "x_b"} : (!fir.ptr<!s1>) -> !fir.ref<f32>
  return
}

// -----

// EQUIVALENCE of arrays of the same sequence type at different offsets:
//   type(s1) :: x(2), y(2)   ! components a, b
//   real :: z(4)
//   equivalence (x(1), z(1)), (y(1), z(2))
// x(1)%b and y(1)%a are the same memory.

// CHECK-LABEL: Testing : "_QPequiv_shift"
// CHECK-DAG: x_b#0 <-> y_a#0: MayAlias

!s1 = !fir.type<_QFequiv_shiftTs1,sequence{a:f32,b:f32}>
func.func @_QPequiv_shift() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %0 = fir.alloca !fir.array<20xi8> {uniq_name = "_QFequiv_shiftEx"}
  %sh = fir.shape %c2 : (index) -> !fir.shape<1>
  %1 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  %2 = fir.convert %1 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2x!s1>>
  %x = fir.declare %2(%sh) storage(%0[0]) uniq_name("_QFequiv_shiftEx") : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, !fir.ref<!fir.array<20xi8>>) -> !fir.ptr<!fir.array<2x!s1>>
  %3 = fir.coordinate_of %0, %c4 : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2x!s1>>
  %y = fir.declare %4(%sh) storage(%0[4]) uniq_name("_QFequiv_shiftEy") : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, !fir.ref<!fir.array<20xi8>>) -> !fir.ptr<!fir.array<2x!s1>>
  %x1 = fir.array_coor %x(%sh) %c1 : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, index) -> !fir.ref<!s1>
  %x_b = fir.coordinate_of %x1, b {test.ptr = "x_b"} : (!fir.ref<!s1>) -> !fir.ref<f32>
  %y1 = fir.array_coor %y(%sh) %c1 : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, index) -> !fir.ref<!s1>
  %y_a = fir.coordinate_of %y1, a {test.ptr = "y_a"} : (!fir.ref<!s1>) -> !fir.ref<f32>
  return
}

// -----

// The same EQUIVALENCE as in _QPequiv_shift after inlining, so that the
// declarations of x and y are compared by the scope-aware refinement.

// CHECK-LABEL: Testing : "_QPequiv_shift_scopes"
// CHECK-DAG: x_b#0 <-> y_a#0: MayAlias

!s1 = !fir.type<_QFequiv_shift_scopesTs1,sequence{a:f32,b:f32}>
func.func @_QPequiv_shift_scopes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %ds = fir.dummy_scope : !fir.dscope
  %0 = fir.alloca !fir.array<20xi8> {uniq_name = "_QFequiv_shift_scopesEx"}
  %sh = fir.shape %c2 : (index) -> !fir.shape<1>
  %1 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  %2 = fir.convert %1 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2x!s1>>
  %x = fir.declare %2(%sh) storage(%0[0]) uniq_name("_QFequiv_shift_scopesEx") : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, !fir.ref<!fir.array<20xi8>>) -> !fir.ptr<!fir.array<2x!s1>>
  %3 = fir.coordinate_of %0, %c4 : (!fir.ref<!fir.array<20xi8>>, index) -> !fir.ref<i8>
  %4 = fir.convert %3 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2x!s1>>
  %y = fir.declare %4(%sh) storage(%0[4]) uniq_name("_QFequiv_shift_scopesEy") : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, !fir.ref<!fir.array<20xi8>>) -> !fir.ptr<!fir.array<2x!s1>>
  %ds_inlined = fir.dummy_scope : !fir.dscope
  %x1 = fir.array_coor %x(%sh) %c1 : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, index) -> !fir.ref<!s1>
  %x_b = fir.coordinate_of %x1, b {test.ptr = "x_b"} : (!fir.ref<!s1>) -> !fir.ref<f32>
  %y1 = fir.array_coor %y(%sh) %c1 : (!fir.ptr<!fir.array<2x!s1>>, !fir.shape<1>, index) -> !fir.ref<!s1>
  %y_a = fir.coordinate_of %y1, a {test.ptr = "y_a"} : (!fir.ref<!s1>) -> !fir.ref<f32>
  return
}

// -----

// The same SEQUENCE type declared in two scopes is the same Fortran type,
// but two different FIR record types. After inlining, the actual argument
// obj%s (of the caller's type) is accessed through the dummy y (of the
// callee's type): obj%s%a and y%a are the same memory.

// CHECK-LABEL: Testing : "_QPsame_sequence_type"
// CHECK-DAG: x_a#0 <-> y_a#0: MayAlias
// CHECK-DAG: x_a#0 <-> y_b#0: MayAlias
// CHECK-DAG: y_a#0 <-> y_b#0: NoAlias

!sa = !fir.type<_QFcallerTs,sequence{a:f32,b:f32}>
!sb = !fir.type<_QFcalleeTs,sequence{a:f32,b:f32}>
!w = !fir.type<_QFcallerTw{s:!sa,z:f32}>
func.func @_QPsame_sequence_type() {
  %caller_scope = fir.dummy_scope : !fir.dscope
  %0 = fir.alloca !w {bindc_name = "obj", uniq_name = "_QFEobj"}
  %obj = fir.declare %0 uniq_name("_QFEobj") : (!fir.ref<!w>) -> !fir.ref<!w>
  %x = fir.coordinate_of %obj, s : (!fir.ref<!w>) -> !fir.ref<!sa>
  %x_a = fir.coordinate_of %obj, s, a {test.ptr = "x_a"} : (!fir.ref<!w>) -> !fir.ref<f32>
  %actual = fir.convert %x : (!fir.ref<!sa>) -> !fir.ref<!sb>
  %ds = fir.dummy_scope : !fir.dscope
  %y = fir.declare %actual dummy_scope %ds arg 1 uniq_name("_QFcalleeEy") : (!fir.ref<!sb>, !fir.dscope) -> !fir.ref<!sb>
  %y_a = fir.coordinate_of %y, a {test.ptr = "y_a"} : (!fir.ref<!sb>) -> !fir.ref<f32>
  %y_b = fir.coordinate_of %y, b {test.ptr = "y_b"} : (!fir.ref<!sb>) -> !fir.ref<f32>
  return
}

// -----

// A first component named after its type is not a parent component:
//   use ma, only: ta => t, shift
//   use mb, only: tb => t
//   type outer
//     type(ta) :: t
//     type(tb) :: other
//   end type
//   type(outer) :: obj
//   call shift(obj%other)
// where modules ma and mb both define
//   type t
//     sequence
//     real :: a(4)
//   end type
// and ma also contains
//   subroutine shift(x)
//     type(t) :: x
//     x%a(2:4) = x%a(1:3)
//   end subroutine
// The two definitions of t describe the same type (F2023 7.5.2.4 p2). After
// inlining, the actual argument obj%other (of type tb) is accessed through the
// dummy x (of type ta): obj%other%a and x%a are the same memory.

// CHECK-LABEL: Testing : "_QPsequence_component"
// CHECK-DAG: obj_t_a#0 <-> x_a_i#0: NoAlias
// CHECK-DAG: obj_other_a#0 <-> x_a_i#0: MayAlias
// CHECK-DAG: x_a_i#0 <-> x_a_j#0: MayAlias

!ta = !fir.type<_QMmaTt,sequence{a:!fir.array<4xf32>}>
!tb = !fir.type<_QMmbTt,sequence{a:!fir.array<4xf32>}>
!outer = !fir.type<_QFsequence_componentTouter{t:!ta,other:!tb}>
fir.type_info @_QMmaTt : !ta
fir.type_info @_QMmbTt : !tb
fir.type_info @_QFsequence_componentTouter : !outer
func.func @_QPsequence_component(%i: index, %j: index) {
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c4 : (index) -> !fir.shape<1>
  %caller_scope = fir.dummy_scope : !fir.dscope
  %0 = fir.alloca !outer {bindc_name = "obj", uniq_name = "_QFsequence_componentEobj"}
  %obj = fir.declare %0 uniq_name("_QFsequence_componentEobj") : (!fir.ref<!outer>) -> !fir.ref<!outer>
  %ta = fir.coordinate_of %obj, t, a : (!fir.ref<!outer>) -> !fir.ref<!fir.array<4xf32>>
  %obj_t_a = fir.array_coor %ta(%sh) %i {test.ptr = "obj_t_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %other = fir.coordinate_of %obj, other : (!fir.ref<!outer>) -> !fir.ref<!tb>
  %oa = fir.coordinate_of %other, a : (!fir.ref<!tb>) -> !fir.ref<!fir.array<4xf32>>
  %obj_other_a = fir.array_coor %oa(%sh) %i {test.ptr = "obj_other_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %actual = fir.convert %other : (!fir.ref<!tb>) -> !fir.ref<!ta>
  %ds = fir.dummy_scope : !fir.dscope
  %x = fir.declare %actual dummy_scope %ds arg 1 uniq_name("_QMmaFshiftEx") : (!fir.ref<!ta>, !fir.dscope) -> !fir.ref<!ta>
  %xa = fir.coordinate_of %x, a : (!fir.ref<!ta>) -> !fir.ref<!fir.array<4xf32>>
  %x_a_i = fir.array_coor %xa(%sh) %i {test.ptr = "x_a_i"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %x_a_j = fir.array_coor %xa(%sh) %j {test.ptr = "x_a_j"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return
}

// -----

// The same with BIND(C) types t, which are the same type too, but are not
// SEQUENCE types in FIR. Modules ma and mb both use c_float from iso_c_binding
// and define
//   type, bind(c) :: t
//     real(c_float) :: a(4)
//   end type

// CHECK-LABEL: Testing : "_QPbind_c_component"
// CHECK-DAG: obj_t_a#0 <-> x_a_i#0: NoAlias
// CHECK-DAG: obj_other_a#0 <-> x_a_i#0: MayAlias
// CHECK-DAG: x_a_i#0 <-> x_a_j#0: MayAlias

!ta = !fir.type<_QMmaTt{a:!fir.array<4xf32>}>
!tb = !fir.type<_QMmbTt{a:!fir.array<4xf32>}>
!outer = !fir.type<_QFbind_c_componentTouter{t:!ta,other:!tb}>
fir.type_info @_QMmaTt : !ta
fir.type_info @_QMmbTt : !tb
fir.type_info @_QFbind_c_componentTouter : !outer
func.func @_QPbind_c_component(%i: index, %j: index) {
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c4 : (index) -> !fir.shape<1>
  %caller_scope = fir.dummy_scope : !fir.dscope
  %0 = fir.alloca !outer {bindc_name = "obj", uniq_name = "_QFbind_c_componentEobj"}
  %obj = fir.declare %0 uniq_name("_QFbind_c_componentEobj") : (!fir.ref<!outer>) -> !fir.ref<!outer>
  %ta = fir.coordinate_of %obj, t, a : (!fir.ref<!outer>) -> !fir.ref<!fir.array<4xf32>>
  %obj_t_a = fir.array_coor %ta(%sh) %i {test.ptr = "obj_t_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %other = fir.coordinate_of %obj, other : (!fir.ref<!outer>) -> !fir.ref<!tb>
  %oa = fir.coordinate_of %other, a : (!fir.ref<!tb>) -> !fir.ref<!fir.array<4xf32>>
  %obj_other_a = fir.array_coor %oa(%sh) %i {test.ptr = "obj_other_a"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %actual = fir.convert %other : (!fir.ref<!tb>) -> !fir.ref<!ta>
  %ds = fir.dummy_scope : !fir.dscope
  %x = fir.declare %actual dummy_scope %ds arg 1 uniq_name("_QMmaFshiftEx") : (!fir.ref<!ta>, !fir.dscope) -> !fir.ref<!ta>
  %xa = fir.coordinate_of %x, a : (!fir.ref<!ta>) -> !fir.ref<!fir.array<4xf32>>
  %x_a_i = fir.array_coor %xa(%sh) %i {test.ptr = "x_a_i"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %x_a_j = fir.array_coor %xa(%sh) %j {test.ptr = "x_a_j"} : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return
}

// -----

// Sequence association of an element of a component after inlining:
//   type r
//     real :: c(10), d(10)
//   end type
//   call foo(obj%c(1), n)     ! subroutine foo(x, n); real :: x(n)
//   ... obj%d(j) ...
// x(i) is obj%c(i), which cannot overlap obj%d(j).

// CHECK-LABEL: Testing : "_QPsequence_association"
// CHECK-DAG: x_i#0 <-> obj_d_j#0: NoAlias

!r = !fir.type<_QMmTr{c:!fir.array<10xf32>,d:!fir.array<10xf32>}>
func.func @_QPsequence_association(%n: index, %i: index, %j: index) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %sh10 = fir.shape %c10 : (index) -> !fir.shape<1>
  %ds_caller = fir.dummy_scope : !fir.dscope
  %0 = fir.alloca !r {bindc_name = "obj", uniq_name = "_QFsequence_associationEobj"}
  %obj = fir.declare %0 uniq_name("_QFsequence_associationEobj") : (!fir.ref<!r>) -> !fir.ref<!r>
  %c = fir.coordinate_of %obj, c : (!fir.ref<!r>) -> !fir.ref<!fir.array<10xf32>>
  %c_1 = fir.array_coor %c(%sh10) %c1 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %actual = fir.convert %c_1 : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xf32>>
  %ds = fir.dummy_scope : !fir.dscope
  %shn = fir.shape %n : (index) -> !fir.shape<1>
  %x = fir.declare %actual(%shn) dummy_scope %ds arg 1 uniq_name("_QFfooEx") : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xf32>>
  %x_i = fir.array_coor %x(%shn) %i {test.ptr = "x_i"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %d = fir.coordinate_of %obj, d : (!fir.ref<!r>) -> !fir.ref<!fir.array<10xf32>>
  %d_j = fir.array_coor %d(%sh10) %j {test.ptr = "obj_d_j"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return
}

// -----

// Views of the component section x%in of an array x of type outer:
// slice_q and coor_q are x(k)%in%q. The component path of the slice is not
// recorded, so the views are not disambiguated from x(k)%j.

// CHECK-LABEL: Testing : "_QPslice_path"
// CHECK-DAG: direct_q#0 <-> direct_j#0: NoAlias
// CHECK-DAG: direct_q#0 <-> slice_q#0: MayAlias
// CHECK-DAG: direct_j#0 <-> slice_q#0: MayAlias
// CHECK-DAG: direct_p#0 <-> slice_q#0: NoAlias
// CHECK-DAG: direct_q#0 <-> coor_q#0: MayAlias
// CHECK-DAG: direct_j#0 <-> coor_q#0: MayAlias
// CHECK-DAG: direct_p#0 <-> coor_q#0: NoAlias

!inner = !fir.type<_QMmTinner{p:f32,q:f32}>
!outer = !fir.type<_QMmTouter{in:!inner,j:f32}>
func.func @_QPslice_path(%arg0: !fir.ref<!fir.array<10x!outer>> {fir.bindc_name = "x"}, %k: index) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %sh = fir.shape %c10 : (index) -> !fir.shape<1>
  %ds = fir.dummy_scope : !fir.dscope
  %x = fir.declare %arg0(%sh) dummy_scope %ds arg 1 uniq_name("_QFslice_pathEx") : (!fir.ref<!fir.array<10x!outer>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<10x!outer>>
  %elt = fir.array_coor %x(%sh) %k : (!fir.ref<!fir.array<10x!outer>>, !fir.shape<1>, index) -> !fir.ref<!outer>
  %direct_q = fir.coordinate_of %elt, in, q {test.ptr = "direct_q"} : (!fir.ref<!outer>) -> !fir.ref<f32>
  %direct_j = fir.coordinate_of %elt, j {test.ptr = "direct_j"} : (!fir.ref<!outer>) -> !fir.ref<f32>
  %direct_p = fir.coordinate_of %elt, in, p {test.ptr = "direct_p"} : (!fir.ref<!outer>) -> !fir.ref<f32>
  %field = fir.field_index in, !outer
  %slice = fir.slice %c1, %c10, %c1 path %field : (index, index, index, !fir.field) -> !fir.slice<1>
  %box = fir.embox %x(%sh) [%slice] : (!fir.ref<!fir.array<10x!outer>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10x!inner>>
  %selt = fir.array_coor %box %k : (!fir.box<!fir.array<10x!inner>>, index) -> !fir.ref<!inner>
  %slice_q = fir.coordinate_of %selt, q {test.ptr = "slice_q"} : (!fir.ref<!inner>) -> !fir.ref<f32>
  %celt = fir.array_coor %x(%sh) [%slice] %k : (!fir.ref<!fir.array<10x!outer>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<!inner>
  %coor_q = fir.coordinate_of %celt, q {test.ptr = "coor_q"} : (!fir.ref<!inner>) -> !fir.ref<f32>
  return
}

// -----

// A pointer component descriptor reinterpreted as a descriptor of data of
// an unrelated type.

// CHECK-LABEL: Testing : "_QPdescriptor_cast"
// CHECK-DAG: direct_a#0 <-> cast_y#0: MayAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!u = !fir.type<_QMmTu{x:f32,y:f32}>
!h = !fir.type<_QMmTh{p:!fir.box<!fir.ptr<!t1>>,k:f32}>
func.func @_QPdescriptor_cast() {
  %0 = fir.alloca !h {bindc_name = "obj", uniq_name = "_QFdescriptor_castEobj"}
  %obj = fir.declare %0 uniq_name("_QFdescriptor_castEobj") : (!fir.ref<!h>) -> !fir.ref<!h>
  %p = fir.coordinate_of %obj, p : (!fir.ref<!h>) -> !fir.ref<!fir.box<!fir.ptr<!t1>>>
  %pbox = fir.load %p : !fir.ref<!fir.box<!fir.ptr<!t1>>>
  %paddr = fir.box_addr %pbox : (!fir.box<!fir.ptr<!t1>>) -> !fir.ptr<!t1>
  %direct_a = fir.coordinate_of %paddr, a {test.ptr = "direct_a"} : (!fir.ptr<!t1>) -> !fir.ref<!fir.array<4xf32>>
  %cast = fir.convert %p : (!fir.ref<!fir.box<!fir.ptr<!t1>>>) -> !fir.ref<!fir.box<!fir.ptr<!u>>>
  %cbox = fir.load %cast : !fir.ref<!fir.box<!fir.ptr<!u>>>
  %caddr = fir.box_addr %cbox : (!fir.box<!fir.ptr<!u>>) -> !fir.ptr<!u>
  %cast_y = fir.coordinate_of %caddr, y {test.ptr = "cast_y"} : (!fir.ptr<!u>) -> !fir.ref<f32>
  return
}

// -----

// Allocatable components of an extended type:
//   type base
//     real :: v(4)
//   end type
//   type, extends(base) :: child
//     type(base), allocatable :: a, b
//   end type
//   type(child) :: x
//   x%a%v(2:4) = x%a%v(1:3)
// The access path of both sections is [a, AllocDeref, v]: component a of x,
// the allocated object x%a, and component v of x%a. Component a of x is
// disjoint from the parent component of x, which contains x%v, but v here is
// a component of x%a, not of x, so this does not prove the sections disjoint.

// CHECK-LABEL: Testing : "_QPalloc_component"
// CHECK-DAG: a_v_lo#0 <-> a_v_hi#0: MayAlias
// CHECK-DAG: a_v_lo#0 <-> b_v#0: NoAlias
// CHECK-DAG: a_v_lo#0 <-> x_v#0: NoAlias

!base = !fir.type<_QMmTbase{v:!fir.array<4xf32>}>
!child = !fir.type<_QMmTchild{base:!base,a:!fir.box<!fir.heap<!base>>,b:!fir.box<!fir.heap<!base>>}>
fir.type_info @_QMmTbase : !base
fir.type_info @_QMmTchild extends !base : !child
func.func @_QPalloc_component(%arg0: !fir.ref<!child> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sh4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %ds = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFalloc_componentEx") : (!fir.ref<!child>, !fir.dscope) -> (!fir.ref<!child>, !fir.ref<!child>)
  %a = hlfir.designate %x#0{"a"} fortran_attrs<allocatable> : (!fir.ref<!child>) -> !fir.ref<!fir.box<!fir.heap<!base>>>
  %abox = fir.load %a : !fir.ref<!fir.box<!fir.heap<!base>>>
  %aaddr = fir.box_addr %abox : (!fir.box<!fir.heap<!base>>) -> !fir.heap<!base>
  %a_v_lo = hlfir.designate %aaddr{"v"} <%sh4> (%c1:%c3:%c1) shape %sh3 {test.ptr = "a_v_lo"} : (!fir.heap<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %a_v_hi = hlfir.designate %aaddr{"v"} <%sh4> (%c2:%c4:%c1) shape %sh3 {test.ptr = "a_v_hi"} : (!fir.heap<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %b = hlfir.designate %x#0{"b"} fortran_attrs<allocatable> : (!fir.ref<!child>) -> !fir.ref<!fir.box<!fir.heap<!base>>>
  %bbox = fir.load %b : !fir.ref<!fir.box<!fir.heap<!base>>>
  %baddr = fir.box_addr %bbox : (!fir.box<!fir.heap<!base>>) -> !fir.heap<!base>
  %b_v = hlfir.designate %baddr{"v"} shape %sh4 {test.ptr = "b_v"} : (!fir.heap<!base>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  %parent = hlfir.designate %x#0{"base"} : (!fir.ref<!child>) -> !fir.ref<!base>
  %x_v = hlfir.designate %parent{"v"} shape %sh4 {test.ptr = "x_v"} : (!fir.ref<!base>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  return
}

// -----

// A pointer component that may be associated with an allocatable component
// of the same object:
//   type, extends(base) :: child
//     type(base), allocatable :: a
//     type(base), pointer :: p
//   end type
//   type(child), target :: x
//   x%a%v(2:4) = x%p%v(1:3)

// CHECK-LABEL: Testing : "_QPtarget_component"
// CHECK-DAG: p_v#0 <-> a_v#0: MayAlias

!base = !fir.type<_QMmTbase{v:!fir.array<4xf32>}>
!child = !fir.type<_QMmTchild{base:!base,a:!fir.box<!fir.heap<!base>>,p:!fir.box<!fir.ptr<!base>>}>
fir.type_info @_QMmTbase : !base
fir.type_info @_QMmTchild extends !base : !child
func.func @_QPtarget_component(%arg0: !fir.ref<!child> {fir.bindc_name = "x", fir.target}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sh4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %ds = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFtarget_componentEx") fortran_attrs<target> : (!fir.ref<!child>, !fir.dscope) -> (!fir.ref<!child>, !fir.ref<!child>)
  %p = hlfir.designate %x#0{"p"} fortran_attrs<pointer> : (!fir.ref<!child>) -> !fir.ref<!fir.box<!fir.ptr<!base>>>
  %pbox = fir.load %p : !fir.ref<!fir.box<!fir.ptr<!base>>>
  %paddr = fir.box_addr %pbox : (!fir.box<!fir.ptr<!base>>) -> !fir.ptr<!base>
  %p_v = hlfir.designate %paddr{"v"} <%sh4> (%c1:%c3:%c1) shape %sh3 {test.ptr = "p_v"} : (!fir.ptr<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %a = hlfir.designate %x#0{"a"} fortran_attrs<allocatable> : (!fir.ref<!child>) -> !fir.ref<!fir.box<!fir.heap<!base>>>
  %abox = fir.load %a : !fir.ref<!fir.box<!fir.heap<!base>>>
  %aaddr = fir.box_addr %abox : (!fir.box<!fir.heap<!base>>) -> !fir.heap<!base>
  %a_v = hlfir.designate %aaddr{"v"} <%sh4> (%c2:%c4:%c1) shape %sh3 {test.ptr = "a_v"} : (!fir.heap<!base>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  return
}

// -----

// A recursive allocatable component:
//   type t
//     real :: v(4)
//     type(t), allocatable :: next
//   end type
//   type(t) :: x
//   x%next%v(2:4) = x%next%v(1:3)
// The access path of both sections is [next, AllocDeref, v]: component next
// of x, the allocated object x%next, and component v of x%next. The steps
// next and v select different components of type t, but of different objects
// (x and x%next), so they do not prove the sections disjoint.

// CHECK-LABEL: Testing : "_QPrecursive_component"
// CHECK-DAG: next_v_lo#0 <-> next_v_hi#0: MayAlias
// CHECK-DAG: next_v_lo#0 <-> x_v#0: NoAlias

!t = !fir.type<_QMmTt{v:!fir.array<4xf32>,next:!fir.box<!fir.heap<!fir.type<_QMmTt>>>}>
func.func @_QPrecursive_component(%arg0: !fir.ref<!t> {fir.bindc_name = "x"}) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sh4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %ds = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFrecursive_componentEx") : (!fir.ref<!t>, !fir.dscope) -> (!fir.ref<!t>, !fir.ref<!t>)
  %next = hlfir.designate %x#0{"next"} fortran_attrs<allocatable> : (!fir.ref<!t>) -> !fir.ref<!fir.box<!fir.heap<!t>>>
  %nbox = fir.load %next : !fir.ref<!fir.box<!fir.heap<!t>>>
  %naddr = fir.box_addr %nbox : (!fir.box<!fir.heap<!t>>) -> !fir.heap<!t>
  %next_v_lo = hlfir.designate %naddr{"v"} <%sh4> (%c1:%c3:%c1) shape %sh3 {test.ptr = "next_v_lo"} : (!fir.heap<!t>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %next_v_hi = hlfir.designate %naddr{"v"} <%sh4> (%c2:%c4:%c1) shape %sh3 {test.ptr = "next_v_hi"} : (!fir.heap<!t>, !fir.shape<1>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
  %x_v = hlfir.designate %x#0{"v"} shape %sh4 {test.ptr = "x_v"} : (!fir.ref<!t>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf32>>
  return
}
