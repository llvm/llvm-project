// Tests for access paths in fir::AliasAnalysis through views that change the
// type of the addressed object: CLASS views of objects of extended types,
// SELECT TYPE, storage association, component paths of array slices, and
// descriptor casts.

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
// sel%a and this%a are the same component. this%a and sel%b are disjoint,
// but the analysis does not relate the components of the two views when
// the dynamic type of the dummy is unknown.

// CHECK-LABEL: Testing : "_QPsel"
// CHECK-DAG: this_a#0 <-> sel_a#0: MayAlias
// CHECK-DAG: this_a#0 <-> sel_b#0: MayAlias
// CHECK-DAG: sel_a#0 <-> sel_b#0: NoAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!t2 = !fir.type<_QMmTt2{t1:!t1,b:f32}>
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
// x%b and y%d are the same memory. The component paths of the two variables
// cannot be related, so even x%a and x%b are not disambiguated.

// CHECK-LABEL: Testing : "_QPequiv"
// CHECK-DAG: x_a#0 <-> y_d#0: MayAlias
// CHECK-DAG: x_a#0 <-> x_b#0: MayAlias
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

// A component reinterpreted as an object of a larger type. The
// reinterpreted object overlaps the sibling component that follows: pun_r
// and obj%d are the same memory.

// CHECK-LABEL: Testing : "_QPpun"
// CHECK-DAG: pun_r#0 <-> direct_d#0: MayAlias

!r = !fir.type<_QMmTr{c:!fir.array<2xf32>,d:f32}>
!s = !fir.type<_QMmTs{p:f32,q:f32,r:f32}>
func.func @_QPpun() {
  %0 = fir.alloca !r {bindc_name = "obj", uniq_name = "_QFpunEobj"}
  %obj = fir.declare %0 uniq_name("_QFpunEobj") : (!fir.ref<!r>) -> !fir.ref<!r>
  %c = fir.coordinate_of %obj, c : (!fir.ref<!r>) -> !fir.ref<!fir.array<2xf32>>
  %pun = fir.convert %c : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<!s>
  %pun_r = fir.coordinate_of %pun, r {test.ptr = "pun_r"} : (!fir.ref<!s>) -> !fir.ref<f32>
  %direct_d = fir.coordinate_of %obj, d {test.ptr = "direct_d"} : (!fir.ref<!r>) -> !fir.ref<f32>
  return
}

// -----

// Views of the component section x%in of an array x of type outer:
// slice_q and coor_q are x(k)%in%q.

// CHECK-LABEL: Testing : "_QPslice_path"
// CHECK-DAG: direct_q#0 <-> direct_j#0: NoAlias
// CHECK-DAG: direct_q#0 <-> slice_q#0: MayAlias
// CHECK-DAG: direct_j#0 <-> slice_q#0: NoAlias
// CHECK-DAG: direct_p#0 <-> slice_q#0: NoAlias
// CHECK-DAG: direct_q#0 <-> coor_q#0: MayAlias
// CHECK-DAG: direct_j#0 <-> coor_q#0: NoAlias
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

// The same EQUIVALENCE as in _QPequiv_shift after inlining, so that the
// declarations of x and y are compared by the scope-aware refinement. The
// views crossed after the declarations of x and y reinterpret the storage
// for them too.

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

// The two-element array component c viewed as a three-element array: the
// third element of the view, view_3, is obj%d.

// CHECK-LABEL: Testing : "_QPview_element"
// CHECK-DAG: view_3#0 <-> direct_d#0: MayAlias
// CHECK-DAG: direct_c2#0 <-> direct_d#0: NoAlias

!r = !fir.type<_QMmTr{c:!fir.array<2xf32>,d:f32}>
func.func @_QPview_element() {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %sh2 = fir.shape %c2 : (index) -> !fir.shape<1>
  %sh3 = fir.shape %c3 : (index) -> !fir.shape<1>
  %0 = fir.alloca !r {bindc_name = "obj", uniq_name = "_QFview_elementEobj"}
  %obj = fir.declare %0 uniq_name("_QFview_elementEobj") : (!fir.ref<!r>) -> !fir.ref<!r>
  %c = fir.coordinate_of %obj, c : (!fir.ref<!r>) -> !fir.ref<!fir.array<2xf32>>
  %view = fir.convert %c : (!fir.ref<!fir.array<2xf32>>) -> !fir.ref<!fir.array<3xf32>>
  %view_3 = fir.array_coor %view(%sh3) %c3 {test.ptr = "view_3"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %direct_c2 = fir.array_coor %c(%sh2) %c2 {test.ptr = "direct_c2"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %direct_d = fir.coordinate_of %obj, d {test.ptr = "direct_d"} : (!fir.ref<!r>) -> !fir.ref<f32>
  return
}

// -----

// The component c viewed as an object of a larger type: the whole view,
// view, overlaps obj%d.

// CHECK-LABEL: Testing : "_QPview_scalar"
// CHECK-DAG: view#0 <-> direct_d#0: MayAlias

!r = !fir.type<_QMmTr{c:f32,d:f32}>
!bigger = !fir.type<_QMmTbigger{x:f32,y:f32}>
func.func @_QPview_scalar() {
  %0 = fir.alloca !r {bindc_name = "obj", uniq_name = "_QFview_scalarEobj"}
  %obj = fir.declare %0 uniq_name("_QFview_scalarEobj") : (!fir.ref<!r>) -> !fir.ref<!r>
  %c = fir.coordinate_of %obj, c : (!fir.ref<!r>) -> !fir.ref<f32>
  %view = fir.convert %c {test.ptr = "view"} : (!fir.ref<f32>) -> !fir.ref<!bigger>
  %direct_d = fir.coordinate_of %obj, d {test.ptr = "direct_d"} : (!fir.ref<!r>) -> !fir.ref<f32>
  return
}

// -----

// Sequence association of an element of an array component with an
// explicit-shape dummy array, after inlining:
//   type r
//     real :: c(2), d
//   end type
//   call foo(obj%c(2))
//   subroutine foo(x)
//     real :: x(2)
// x(2) is obj%d.

// CHECK-LABEL: Testing : "_QPsequence_association"
// CHECK-DAG: x_2#0 <-> obj_d#0: MayAlias

!r = !fir.type<_QMmTr{c:!fir.array<2xf32>,d:f32}>
func.func @_QPsequence_association() {
  %c2 = arith.constant 2 : index
  %0 = fir.alloca !r {bindc_name = "obj", uniq_name = "_QFsequence_associationEobj"}
  %obj:2 = hlfir.declare %0 uniq_name("_QFsequence_associationEobj") : (!fir.ref<!r>) -> (!fir.ref<!r>, !fir.ref<!r>)
  %sh = fir.shape %c2 : (index) -> !fir.shape<1>
  %c_2 = hlfir.designate %obj#0{"c"} <%sh> (%c2) : (!fir.ref<!r>, !fir.shape<1>, index) -> !fir.ref<f32>
  %actual = fir.convert %c_2 : (!fir.ref<f32>) -> !fir.ref<!fir.array<2xf32>>
  %ds = fir.dummy_scope : !fir.dscope
  %x:2 = hlfir.declare %actual(%sh) dummy_scope %ds arg 1 uniq_name("_QFfooEx") : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
  %x_2 = hlfir.designate %x#0 (%c2) {test.ptr = "x_2"} : (!fir.ref<!fir.array<2xf32>>, index) -> !fir.ref<f32>
  %obj_d = hlfir.designate %obj#0{"d"} {test.ptr = "obj_d"} : (!fir.ref<!r>) -> !fir.ref<f32>
  return
}

// -----

// A pointer component descriptor reinterpreted as a derived type object:
// view_word is the first word of the descriptor of obj%p.

// CHECK-LABEL: Testing : "_QPdescriptor_view"
// CHECK-DAG: desc#0 <-> view_word#0: MayAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!w = !fir.type<_QMmTw{word:i64}>
!h = !fir.type<_QMmTh{p:!fir.box<!fir.ptr<!t1>>,k:f32}>
func.func @_QPdescriptor_view() {
  %0 = fir.alloca !h {bindc_name = "obj", uniq_name = "_QFdescriptor_viewEobj"}
  %obj = fir.declare %0 uniq_name("_QFdescriptor_viewEobj") : (!fir.ref<!h>) -> !fir.ref<!h>
  %desc = fir.coordinate_of %obj, p {test.ptr = "desc"} : (!fir.ref<!h>) -> !fir.ref<!fir.box<!fir.ptr<!t1>>>
  %view = fir.convert %desc : (!fir.ref<!fir.box<!fir.ptr<!t1>>>) -> !fir.ref<!w>
  %view_word = fir.coordinate_of %view, word {test.ptr = "view_word"} : (!fir.ref<!w>) -> !fir.ref<i64>
  return
}

// -----

// A pointer component descriptor reinterpreted as an integer: word is the
// first word of the descriptor of obj%p. It is reached through another
// access to obj%p, so that it does not trivially address the same memory as
// desc.

// CHECK-LABEL: Testing : "_QPdescriptor_word"
// CHECK-DAG: desc#0 <-> word#0: MayAlias

!t1 = !fir.type<_QMmTt1{a:!fir.array<4xf32>}>
!h = !fir.type<_QMmTh{p:!fir.box<!fir.ptr<!t1>>,k:f32}>
func.func @_QPdescriptor_word() {
  %0 = fir.alloca !h {bindc_name = "obj", uniq_name = "_QFdescriptor_wordEobj"}
  %obj = fir.declare %0 uniq_name("_QFdescriptor_wordEobj") : (!fir.ref<!h>) -> !fir.ref<!h>
  %desc = fir.coordinate_of %obj, p {test.ptr = "desc"} : (!fir.ref<!h>) -> !fir.ref<!fir.box<!fir.ptr<!t1>>>
  %desc2 = fir.coordinate_of %obj, p : (!fir.ref<!h>) -> !fir.ref<!fir.box<!fir.ptr<!t1>>>
  %word = fir.convert %desc2 {test.ptr = "word"} : (!fir.ref<!fir.box<!fir.ptr<!t1>>>) -> !fir.ref<i64>
  return
}

// -----

// A box with a byte stride smaller than the size of its elements: the
// first component of its second element, box_a, is the second component of
// the first element of the array, direct_b.

// CHECK-LABEL: Testing : "_QPcreate_box_stride"
// CHECK-DAG: box_a#0 <-> direct_b#0: MayAlias
// CHECK-DAG: direct_a#0 <-> direct_b#0: NoAlias

!s = !fir.type<_QMmTs{a:f32,b:f32}>
func.func @_QPcreate_box_stride() {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %sh = fir.shape %c2 : (index) -> !fir.shape<1>
  %0 = fir.alloca !fir.array<2x!s> {bindc_name = "arr", uniq_name = "_QFcreate_box_strideEarr"}
  %arr = fir.declare %0(%sh) uniq_name("_QFcreate_box_strideEarr") : (!fir.ref<!fir.array<2x!s>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!s>>
  %box = fir.create_box %arr lbs(%c1) extents(%c2) strides(%c4) : (!fir.ref<!fir.array<2x!s>>, index, index, index) -> !fir.box<!fir.array<2x!s>>
  %belt = fir.array_coor %box %c2 : (!fir.box<!fir.array<2x!s>>, index) -> !fir.ref<!s>
  %box_a = fir.coordinate_of %belt, a {test.ptr = "box_a"} : (!fir.ref<!s>) -> !fir.ref<f32>
  %elt = fir.array_coor %arr(%sh) %c1 : (!fir.ref<!fir.array<2x!s>>, !fir.shape<1>, index) -> !fir.ref<!s>
  %direct_a = fir.coordinate_of %elt, a {test.ptr = "direct_a"} : (!fir.ref<!s>) -> !fir.ref<f32>
  %direct_b = fir.coordinate_of %elt, b {test.ptr = "direct_b"} : (!fir.ref<!s>) -> !fir.ref<f32>
  return
}
