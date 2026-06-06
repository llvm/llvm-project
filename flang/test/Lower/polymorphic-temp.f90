! Test creation of temporary from polymorphic enities
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module poly_tmp
  type p1
    integer :: a
  end type

  type, extends(p1) :: p2
    integer :: b
  end type

contains
  subroutine pass_unlimited_poly_1d(x)
    class(*), intent(in) :: x(:)
  end subroutine


  subroutine test_temp_from_intrinsic_spread()
    class(*), pointer :: p
    class(*), pointer :: pa(:)
    allocate(p2::p)
    allocate(p2::pa(10))

    call pass_unlimited_poly_1d(spread(p, dim=1, ncopies=2))
    call pass_unlimited_poly_1d(spread(pa(1), dim=1, ncopies=2))

  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_spread() {
! CHECK: %[[TEMP_RES1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: %[[TEMP_RES0:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "p", uniq_name = "_QMpoly_tmpFtest_temp_from_intrinsic_spreadEp"}
! CHECK: %[[P_DECL:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[PA:.*]] = fir.alloca !fir.class<!fir.ptr<!fir.array<?xnone>>> {bindc_name = "pa", uniq_name = "_QMpoly_tmpFtest_temp_from_intrinsic_spreadEpa"}
! CHECK: %[[PA_DECL:.*]]:2 = hlfir.declare %[[PA]]
! CHECK: fir.call @_FortranAPointerNullifyDerived
! CHECK: fir.call @_FortranAPointerAllocate
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xnone>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! Make sure the fir.embox contains the source_box pointing to the polymoprhic entity
! CHECK: %[[BOX_RES:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) source_box %[[LOAD_P]] : (!fir.heap<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<!fir.ptr<none>>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: fir.store %[[BOX_RES]] to %[[TEMP_RES0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK: %[[RES_BOX_NONE:.*]] = fir.convert %[[TEMP_RES0]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[P_BOX_NONE:.*]] = fir.convert %[[LOAD_P]] : (!fir.class<!fir.ptr<none>>) -> !fir.box<none>
! CHECK: fir.call @_FortranASpread(%[[RES_BOX_NONE]], %[[P_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> ()
! CHECK: %[[LOAD_RES:.*]] = fir.load %[[TEMP_RES0]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK: fir.call @_QMpoly_tmpPpass_unlimited_poly_1d
! CHECK: %[[LOAD_PA:.*]] = fir.load %[[PA_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?xnone>>>>
! CHECK: %[[DESIGNATE_PA_1:.*]] = hlfir.designate %[[LOAD_PA]] (%{{.*}}) : (!fir.class<!fir.ptr<!fir.array<?xnone>>>, index) -> !fir.class<none>
! CHECK: %[[EMBOX_PA_1:.*]] = fir.embox %{{.*}} source_box %[[DESIGNATE_PA_1]] : (!fir.heap<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<none>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: fir.store %[[EMBOX_PA_1]] to %[[TEMP_RES1]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK: %[[RES1_BOX_NONE:.*]] = fir.convert %[[TEMP_RES1]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[PA1_BOX_NONE:.*]] = fir.convert %[[DESIGNATE_PA_1]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: fir.call @_FortranASpread(%[[RES1_BOX_NONE]], %[[PA1_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> ()

  subroutine test_temp_from_intrinsic_reshape(i)
    class(*), allocatable :: a(:,:)
    class(*), intent(in) :: i(20,20)
    allocate(a(10,10), source=reshape(i,[10,10]))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_reshape(
! CHECK-SAME: %[[I:.*]]: !fir.class<!fir.array<20x20xnone>> {fir.bindc_name = "i"}) {
! CHECK: %[[A:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?xnone>>> {bindc_name = "a", uniq_name = "_QMpoly_tmpFtest_temp_from_intrinsic_reshapeEa"}
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[RESHAPE_RES:.*]] = hlfir.reshape %[[I_DECL]]#0 %{{.*}} : (!fir.class<!fir.array<20x20xnone>>, !fir.ref<!fir.array<2xi32>>) -> !hlfir.expr<10x10xnone?>
! CHECK: %[[ASSOC_RES:.*]]:3 = hlfir.associate %[[RESHAPE_RES]]
! CHECK: %[[RANK:.*]] = arith.constant 2 : i32
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[RES_BOX_NONE:.*]] = fir.convert %[[ASSOC_RES]]#1 : (!fir.class<!fir.array<10x10xnone>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %[[RES_BOX_NONE]], %[[RANK]]) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()

  subroutine check(r)
    class(p1) :: r(:)
  end subroutine

  subroutine test_temp_from_intrinsic_pack(i, mask)
    class(p1), intent(in) :: i(20, 20)
    logical, intent(in) :: mask(20, 20)
    call check(pack(i, mask))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_pack(
! CHECK-SAME: %[[I:.*]]: !fir.class<!fir.array<20x20x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "i"}, %[[MASK:.*]]: !fir.ref<!fir.array<20x20x!fir.logical<4>>> {fir.bindc_name = "mask"}) {
! CHECK: %[[TMP_RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK]]
! CHECK: %[[EMBOXED_MASK:.*]] = fir.embox %[[MASK_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<20x20x!fir.logical<4>>>, !fir.shape<2>) -> !fir.box<!fir.array<20x20x!fir.logical<4>>>
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>
! CHECK: %[[EMBOX_RES:.*]] = fir.embox %[[ZERO]](%{{.*}}) source_box %[[I_DECL]]#1 : (!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>, !fir.shape<1>, !fir.class<!fir.array<20x20x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>
! CHECK: fir.store %[[EMBOX_RES]] to %[[TMP_RES]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>>
! CHECK: %[[RES_BOX_NONE:.*]] = fir.convert %[[TMP_RES]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[I_BOX_NONE:.*]] = fir.convert %[[I_DECL]]#1 : (!fir.class<!fir.array<20x20x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.box<none>
! CHECK: %[[MASK_BOX_NONE:.*]] = fir.convert %[[EMBOXED_MASK]] : (!fir.box<!fir.array<20x20x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAPack(%[[RES_BOX_NONE]], %[[I_BOX_NONE]], %[[MASK_BOX_NONE]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()

  subroutine check_rank2(r)
    class(p1), intent(in) :: r(:,:)
  end subroutine

  subroutine test_temp_from_unpack(v, m, f)
    class(p1), intent(in) :: v(:), f(:,:)
    logical, intent(in) :: m(:,:)
    call check_rank2(unpack(v,m,f))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_unpack(
! CHECK-SAME: %[[V:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "v"}, %[[M:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "m"}, %[[F:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "f"}) {
! CHECK: %[[TMP_RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>
! CHECK: %[[F_DECL:.*]]:2 = hlfir.declare %[[F]]
! CHECK: %[[M_DECL:.*]]:2 = hlfir.declare %[[M]]
! CHECK: %[[V_DECL:.*]]:2 = hlfir.declare %[[V]]
! CHECK: %[[TMP_BOX_NONE:.*]] = fir.convert %[[TMP_RES]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[V_BOX_NONE:.*]] = fir.convert %[[V_DECL]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.box<none>
! CHECK: %[[M_BOX_NONE:.*]] = fir.convert %[[M_DECL]]#1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[F_BOX_NONE:.*]] = fir.convert %[[F_DECL]]#1 : (!fir.class<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAUnpack(%[[TMP_BOX_NONE]], %[[V_BOX_NONE]], %[[M_BOX_NONE]], %[[F_BOX_NONE]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()

  subroutine check_cshift(r)
    class(p1) :: r(:)
  end subroutine

  subroutine test_temp_from_intrinsic_cshift(a, shift)
    class(p1), intent(in) :: a(20)
    integer :: shift

    call check(cshift(a, shift))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_cshift(
! CHECK-SAME: %[[ARRAY:.*]]: !fir.class<!fir.array<20x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "a"}, %[[SHIFT:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"}) {
! CHECK: %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[ARRAY]]
! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[SHIFT]]
! CHECK: %[[CSHIFT:.*]] = hlfir.cshift %[[ARRAY_DECL]]#0 %[[SHIFT_DECL]]#0 : (!fir.class<!fir.array<20x!fir.type<_QMpoly_tmpTp1{a:i32}>>>, !fir.ref<i32>) -> !hlfir.expr<20x!fir.type<_QMpoly_tmpTp1{a:i32}>?>

  subroutine test_temp_from_intrinsic_eoshift(a, shift, b)
    class(p1), intent(in) :: a(20)
    class(p1), intent(in) :: b
    integer :: shift

    call check(eoshift(a, shift, b))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_eoshift(
! CHECK-SAME: %[[ARRAY:.*]]: !fir.class<!fir.array<20x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "a"}, %[[SHIFT:.*]]: !fir.ref<i32> {fir.bindc_name = "shift"}, %[[BOUNDARY:.*]]: !fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>> {fir.bindc_name = "b"}) {
! CHECK: %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[ARRAY]]
! CHECK: %[[BOUNDARY_DECL:.*]]:2 = hlfir.declare %[[BOUNDARY]]
! CHECK: %[[SHIFT_DECL:.*]]:2 = hlfir.declare %[[SHIFT]]
! CHECK: %[[EOSHIFT:.*]] = hlfir.eoshift %[[ARRAY_DECL]]#0 %[[SHIFT_DECL]]#0 boundary %[[BOUNDARY_DECL]]#0 : (!fir.class<!fir.array<20x!fir.type<_QMpoly_tmpTp1{a:i32}>>>, !fir.ref<i32>, !fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>>) -> !hlfir.expr<20x!fir.type<_QMpoly_tmpTp1{a:i32}>?>

  subroutine test_temp_from_intrinsic_transfer(source, mold)
    class(p1), intent(in) :: source(:)
    class(p1), intent(in) :: mold(:)
    call check(transfer(source, mold))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_transfer(
! CHECK-SAME: %[[SOURCE:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "source"}, %[[MOLD:.*]]: !fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "mold"}) {
! CHECK: %[[TMP_RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>
! CHECK: %[[MOLD_DECL:.*]]:2 = hlfir.declare %[[MOLD]]
! CHECK: %[[SOURCE_DECL:.*]]:2 = hlfir.declare %[[SOURCE]]
! CHECK: %[[RES_BOX_NONE:.*]] = fir.convert %[[TMP_RES]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[SOURCE_NONE:.*]] = fir.convert %[[SOURCE_DECL]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.box<none>
! CHECK: %[[MOLD_NONE:.*]] = fir.convert %[[MOLD_DECL]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranATransfer(%[[RES_BOX_NONE]], %[[SOURCE_NONE]], %[[MOLD_NONE]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()

  subroutine test_temp_from_intrinsic_transpose(matrix)
    class(p1), intent(in) :: matrix(:,:)
    call check_rank2(transpose(matrix))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_transpose(
! CHECK-SAME: %[[MATRIX:.*]]: !fir.class<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>> {fir.bindc_name = "matrix"}) {
! CHECK: %[[MATRIX_DECL:.*]]:2 = hlfir.declare %[[MATRIX]]
! CHECK: %[[TRANS:.*]] = hlfir.transpose %[[MATRIX_DECL]]#0 : (!fir.class<!fir.array<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>>>) -> !hlfir.expr<?x?x!fir.type<_QMpoly_tmpTp1{a:i32}>?>

  subroutine check_scalar(a)
    class(p1), intent(in) :: a
  end subroutine

  subroutine test_merge_intrinsic(a, b)
    class(p1), intent(in) :: a, b

    call check_scalar(merge(a, b, a%a > b%a))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_merge_intrinsic(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>> {fir.bindc_name = "b"}) {
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[DES_A:.*]] = hlfir.designate %[[A_DECL]]#0{"a"}   : (!fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>>) -> !fir.ref<i32>
! CHECK: %[[LOAD_A1:.*]] = fir.load %[[DES_A]] : !fir.ref<i32>
! CHECK: %[[DES_B:.*]] = hlfir.designate %[[B_DECL]]#0{"a"}   : (!fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>>) -> !fir.ref<i32>
! CHECK: %[[LOAD_A2:.*]] = fir.load %[[DES_B]] : !fir.ref<i32>
! CHECK: %[[CMPI:.*]] = arith.cmpi sgt, %[[LOAD_A1]], %[[LOAD_A2]] : i32
! CHECK: %[[SELECT:.*]] = arith.select %[[CMPI]], %[[A_DECL]]#1, %[[B_DECL]]#1 : !fir.class<!fir.type<_QMpoly_tmpTp1{a:i32}>>

  subroutine test_merge_intrinsic2(a, b, i)
    class(p1), allocatable, intent(in) :: a
    type(p1), allocatable :: b
    integer, intent(in) :: i

    call check_scalar(merge(a, b, i==1))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_merge_intrinsic2(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpoly_tmpTp1{a:i32}>>>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMpoly_tmpTp1{a:i32}>>>> {fir.bindc_name = "b"}, %[[I:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[LOAD_I:.*]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[CMPI:.*]] = arith.cmpi eq, %[[LOAD_I]], %[[C1]] : i32
! CHECK: %[[LOAD_A:.*]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMpoly_tmpTp1{a:i32}>>>>
! CHECK: %[[LOAD_B:.*]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMpoly_tmpTp1{a:i32}>>>>

  subroutine check_unlimited_poly(a)
    class(*), intent(in) :: a
  end subroutine

  subroutine test_merge_intrinsic3(a, b, i)
    class(*), intent(in) :: a, b
    integer, intent(in) :: i

    call check_unlimited_poly(merge(a, b, i==1))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_merge_intrinsic3(
! CHECK-SAME: %[[A:.*]]: !fir.class<none> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.class<none> {fir.bindc_name = "b"}, %[[I:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[V_0:[0-9]+]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[V_1:[0-9]+]] = arith.cmpi eq, %[[V_0]], %[[C1]] : i32
! CHECK: %[[V_2:[0-9]+]] = arith.select %[[V_1]], %[[A_DECL]]#1, %[[B_DECL]]#1 : !fir.class<none>

  subroutine test_merge_intrinsic4(i)
    integer, intent(in) :: i
    class(*), allocatable :: a, b

    call check_unlimited_poly(merge(a, b, i==1))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_merge_intrinsic4(
! CHECK-SAME: %[[I:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
! CHECK: %[[V_0:[0-9]+]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "a", uniq_name = "_QMpoly_tmpFtest_merge_intrinsic4Ea"}
! CHECK: %[[V_1:[0-9]+]] = fir.zero_bits !fir.heap<none>
! CHECK: %[[V_2:[0-9]+]] = fir.embox %[[V_1]] : (!fir.heap<none>) -> !fir.class<!fir.heap<none>>
! CHECK: fir.store %[[V_2]] to %[[V_0]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[V_0]]
! CHECK: %[[V_3:[0-9]+]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "b", uniq_name = "_QMpoly_tmpFtest_merge_intrinsic4Eb"}
! CHECK: %[[V_4:[0-9]+]] = fir.zero_bits !fir.heap<none>
! CHECK: %[[V_5:[0-9]+]] = fir.embox %[[V_4]] : (!fir.heap<none>) -> !fir.class<!fir.heap<none>>
! CHECK: fir.store %[[V_5]] to %[[V_3]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[V_3]]
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[V_8:[0-9]+]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[V_9:[0-9]+]] = arith.cmpi eq, %[[V_8]], %[[C1]] : i32
! CHECK: %[[V_6:[0-9]+]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[V_7:[0-9]+]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK: %[[V_10:[0-9]+]] = arith.select %[[V_9]], %[[V_6]], %[[V_7]] : !fir.class<!fir.heap<none>>

  subroutine test_merge_intrinsic5(i)
    integer, intent(in) :: i
    class(*), pointer :: a, b

    call check_unlimited_poly(merge(a, b, i==1))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_merge_intrinsic5(
! CHECK-SAME: %[[I:.*]]: !fir.ref<i32> {fir.bindc_name = "i"}) {
! CHECK: %[[V_0:[0-9]+]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "a", uniq_name = "_QMpoly_tmpFtest_merge_intrinsic5Ea"}
! CHECK: %[[V_1:[0-9]+]] = fir.zero_bits !fir.ptr<none>
! CHECK: %[[V_2:[0-9]+]] = fir.embox %[[V_1]] : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
! CHECK: fir.store %[[V_2]] to %[[V_0]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[V_0]]
! CHECK: %[[V_3:[0-9]+]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "b", uniq_name = "_QMpoly_tmpFtest_merge_intrinsic5Eb"}
! CHECK: %[[V_4:[0-9]+]] = fir.zero_bits !fir.ptr<none>
! CHECK: %[[V_5:[0-9]+]] = fir.embox %[[V_4]] : (!fir.ptr<none>) -> !fir.class<!fir.ptr<none>>
! CHECK: fir.store %[[V_5]] to %[[V_3]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[B_DECL:.*]]:2 = hlfir.declare %[[V_3]]
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I]]
! CHECK: %[[V_8:[0-9]+]] = fir.load %[[I_DECL]]#0 : !fir.ref<i32>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[V_9:[0-9]+]] = arith.cmpi eq, %[[V_8]], %[[C1]] : i32
! CHECK: %[[V_6:[0-9]+]] = fir.load %[[A_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[V_7:[0-9]+]] = fir.load %[[B_DECL]]#0 : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[V_10:[0-9]+]] = arith.select %[[V_9]], %[[V_6]], %[[V_7]] : !fir.class<!fir.ptr<none>>

end module
