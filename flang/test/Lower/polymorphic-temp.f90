! Test creation of temporary from polymorphic enities
! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

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
    allocate(p2::p)

    call pass_unlimited_poly_1d(spread(p, dim=1, ncopies=2))
  end subroutine

! CHECK-LABEL: func.func @_QMpoly_tmpPtest_temp_from_intrinsic_spread() {
! CHECK: %[[TEMP_RES:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: %[[P:.*]] = fir.alloca !fir.class<!fir.ptr<none>> {bindc_name = "p", uniq_name = "_QMpoly_tmpFtest_temp_from_intrinsic_spreadEp"}
! CHECK: fir.call @_FortranAPointerNullifyDerived
! CHECK: fir.call @_FortranAPointerAllocate
! CHECK: %[[LOAD_P:.*]] = fir.load %[[P]] : !fir.ref<!fir.class<!fir.ptr<none>>>
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[C2:.*]] = arith.constant 2 : i32
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xnone>>
! CHECK: %[[C0:.*]] = arith.constant 0 : index
! CHECK: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! Make sure the fir.embox contains the source_box pointing to the polymoprhic entity
! CHECK: %[[BOX_RES:.*]] = fir.embox %[[ZERO]](%[[SHAPE]]) source_box %[[LOAD_P]] : (!fir.heap<!fir.array<?xnone>>, !fir.shape<1>, !fir.class<!fir.ptr<none>>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK: fir.store %[[BOX_RES]] to %[[TEMP_RES]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK: %[[RES_BOX_NONE:.*]] = fir.convert %[[TEMP_RES]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %[[P_BOX_NONE:.*]] = fir.convert %[[LOAD_P]] : (!fir.class<!fir.ptr<none>>) -> !fir.box<none>
! CHECK: %[[C2_I64:.*]] = fir.convert %[[C2]] : (i32) -> i64
! CHECK: %{{.*}} = fir.call @_FortranASpread(%[[RES_BOX_NONE]], %[[P_BOX_NONE]], %[[C1]], %[[C2_I64]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i64, !fir.ref<i8>, i32) -> none
! CHECK: %[[LOAD_RES:.*]] = fir.load %[[TEMP_RES]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK: %[[RES_ADDR:.*]] = fir.box_addr %[[LOAD_RES]] : (!fir.class<!fir.heap<!fir.array<?xnone>>>) -> !fir.heap<!fir.array<?xnone>>
! CHECK: %[[REBOX:.*]] = fir.rebox %[[LOAD_RES]] : (!fir.class<!fir.heap<!fir.array<?xnone>>>) -> !fir.class<!fir.array<?xnone>>
! CHECK: fir.call @_QMpoly_tmpPpass_unlimited_poly_1d(%[[REBOX]]) {{.*}} : (!fir.class<!fir.array<?xnone>>) -> ()
! CHECK: fir.freemem %[[RES_ADDR]] : !fir.heap<!fir.array<?xnone>>

end module
