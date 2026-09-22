! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Sections of named constants as actual arguments are associated with the
! constant's storage like any other section: a contiguous section is passed
! directly, a noncontiguous section keeps its descriptor (strides intact) for
! an assumed-shape dummy and gets a temporary copy for an explicit-shape
! dummy, and a temporary made from a named constant is never copied back
! into the constant's storage.  Vector-subscripted sections and components
! of sections are not contiguous and go through a temporary.

module ms
  implicit none
  integer, parameter :: sp(6) = [1, 2, 3, 4, 5, 6]
  integer, parameter :: p0(0:*) = [1, 2, 3]
  type t
    integer :: arr(2)
  end type
  type(t), parameter :: pts(3) = [t([1, 2]), t([3, 4]), t([5, 6])]
contains
  subroutine expl3(x)
    integer, intent(in) :: x(3)
  end subroutine
  subroutine asmd(x)
    integer, intent(in) :: x(:)
  end subroutine
end module

! Dynamic-bound strided section to an assumed-shape dummy: the descriptor
! over the named constant's storage is passed directly, strides preserved;
! no copy in either direction.
! CHECK-LABEL: func.func @_QPdyn_section_assumed
! CHECK: %[[A:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[AD:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[SEC:.*]] = hlfir.designate %[[AD]]#0 (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} -> !fir.box<!fir.array<?xi32>>
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.copy_in
! CHECK: fir.call @_QMmsPasmd(%[[SEC]])
! CHECK-NOT: hlfir.copy_out
subroutine dyn_section_assumed(i)
  use ms
  integer :: i
  call asmd(sp(i:6:2))
end subroutine

! The same section to an explicit-shape dummy requires contiguity: a
! temporary copy is made, and nothing is copied back into the named
! constant's storage afterwards (the temporary is simply destroyed).
! CHECK-LABEL: func.func @_QPdyn_section_explicit
! CHECK: %[[B:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[BD:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[BSEC:.*]] = hlfir.designate %[[BD]]#0 (%{{.*}}:%{{.*}}:%{{.*}}) shape
! CHECK: %[[BEXPR:.*]] = hlfir.as_expr %[[BSEC]]
! CHECK: %[[BTMP:.*]]:3 = hlfir.associate %[[BEXPR]]({{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate %[[BTMP]]#1, %[[BTMP]]#2
subroutine dyn_section_explicit(i)
  use ms
  integer :: i
  call expl3(sp(i:6:2))
end subroutine

! Contiguous section with constant bounds to an explicit-shape dummy: the
! constant's storage is passed directly (no outlined constant, no copy).
! CHECK-LABEL: func.func @_QPconst_contig_explicit
! CHECK: %[[C:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[CD:.*]]:2 = hlfir.declare %[[C]]
! CHECK-NOT: _QQro
! CHECK: %[[CSEC:.*]] = hlfir.designate %[[CD]]#0 (%{{.*}}:%{{.*}}:%c1{{.*}}) shape %{{.*}} -> !fir.ref<!fir.array<3xi32>>
! CHECK-NOT: hlfir.as_expr
! CHECK: fir.call @_QMmsPexpl3(%[[CSEC]])
subroutine const_contig_explicit()
  use ms
  call expl3(sp(2:4))
end subroutine

! Strided section with constant bounds to an assumed-shape dummy: descriptor
! over the constant's storage, no copy.
! CHECK-LABEL: func.func @_QPconst_strided_assumed
! CHECK: %[[S:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[SD:.*]]:2 = hlfir.declare %[[S]]
! CHECK-NOT: _QQro
! CHECK: %[[SSEC:.*]] = hlfir.designate %[[SD]]#0 (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} -> !fir.box<!fir.array<3xi32>>
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.copy_in
! CHECK: fir.call @_QMmsPasmd
! CHECK-NOT: hlfir.copy_out
subroutine const_strided_assumed()
  use ms
  call asmd(sp(1:6:2))
end subroutine

! Strided section with constant bounds to an explicit-shape dummy: copied
! from the constant's storage into a temporary; still no copy-out.
! CHECK-LABEL: func.func @_QPconst_strided_explicit
! CHECK: %[[E:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[ED:.*]]:2 = hlfir.declare %[[E]]
! CHECK-NOT: _QQro
! CHECK: %[[ESEC:.*]] = hlfir.designate %[[ED]]#0 (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} -> !fir.box<!fir.array<3xi32>>
! CHECK: %[[EEXPR:.*]] = hlfir.as_expr %[[ESEC]]
! CHECK: %[[ETMP:.*]]:3 = hlfir.associate %[[EEXPR]]({{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3(%[[ETMP]]#0)
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate %[[ETMP]]#1, %[[ETMP]]#2
subroutine const_strided_explicit()
  use ms
  call expl3(sp(1:6:2))
end subroutine

! Vector-subscripted section: the elements are gathered from the constant's
! storage into a temporary (the outlined constant is the subscript vector).
! CHECK-LABEL: func.func @_QPvector_subscript_explicit
! CHECK: %[[V:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[VD:.*]]:2 = hlfir.declare %[[V]]
! CHECK: %[[VGATHER:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<3xi32>
! CHECK: hlfir.designate %[[VD]]#0 (%{{.*}}) : (!fir.ref<!fir.array<6xi32>>, i64) -> !fir.ref<i32>
! CHECK: %[[VTMP:.*]]:3 = hlfir.associate %[[VGATHER]]({{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3(%[[VTMP]]#0)
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate %[[VTMP]]#1, %[[VTMP]]#2
subroutine vector_subscript_explicit()
  use ms
  call expl3(sp([1, 3, 5]))
end subroutine

! Component of a section of a derived-type named constant: noncontiguous,
! so a temporary is made from the constant's storage; no copy-out.
! CHECK-LABEL: func.func @_QPcomponent_of_section
! CHECK: %[[P:.*]] = fir.address_of(@_QMmsECpts)
! CHECK: %[[PD:.*]]:2 = hlfir.declare %[[P]]
! CHECK: %[[PSEC:.*]] = hlfir.designate %[[PD]]#0 (%{{.*}}:%{{.*}}:%c1{{.*}}) shape %{{.*}} -> !fir.ref<!fir.array<3x!fir.type<_QMmsTt{arr:!fir.array<2xi32>}>>>
! CHECK: %[[PCOMP:.*]] = hlfir.designate %[[PSEC]]{"arr"} <%{{.*}}> (%c1{{.*}}) shape %{{.*}} -> !fir.box<!fir.array<3xi32>>
! CHECK: %[[PEXPR:.*]] = hlfir.as_expr %[[PCOMP]]
! CHECK: %[[PTMP:.*]]:3 = hlfir.associate %[[PEXPR]]({{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3(%[[PTMP]]#0)
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate %[[PTMP]]#1, %[[PTMP]]#2
subroutine component_of_section()
  use ms
  call expl3(pts(1:3)%arr(1))
end subroutine

! Omitted-bound section of an implied-shape named constant with a nondefault
! lower bound: its bounds fold, so it is a contiguous section with a static
! extent and the constant's storage is passed directly.
! CHECK-LABEL: func.func @_QPimplied_omitted_bound
! CHECK: %[[I:.*]] = fir.address_of(@_QMmsECp0)
! CHECK: %[[ID:.*]]:2 = hlfir.declare %[[I]]
! CHECK-NOT: _QQro
! CHECK: %[[ISEC:.*]] = hlfir.designate %[[ID]]#0 (%c0{{.*}}:%{{.*}}:%c1{{.*}}) shape %{{.*}} -> !fir.ref<!fir.array<3xi32>>
! CHECK-NOT: hlfir.as_expr
! CHECK: fir.call @_QMmsPexpl3(%[[ISEC]])
subroutine implied_omitted_bound()
  use ms
  call expl3(p0(:))
end subroutine
