! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! A scalar array element associated with an array VALUE dummy via storage
! sequence association (F'2023 15.5.2.12): the temporary that VALUE requires
! must cover the whole sequence the dummy needs, not just the element.  When
! the dummy extents are all known at the call site, the sequence starting at
! the element is viewed as an array of the dummy's shape and that array is
! copied.  Otherwise the whole remaining storage sequence of the base array
! is viewed and copied, mirroring the copy of the whole actual argument that
! is made when the actual argument is an array.

module m
  implicit none
  type tt
    integer :: id
  end type
  type, extends(tt) :: ttx
    integer :: extra
  end type
  type tc
    integer :: a(-1:2)
  end type
  type inner
    integer :: a(-1:0,4:6)
  end type
  type outer
    integer :: guard
    type(inner) :: item(2)
  end type
  type td
    integer, allocatable :: a(:)
  end type
contains
  subroutine byval3(x)
    integer, value :: x(3)
  end subroutine
  subroutine byvaln(n, x)
    integer, intent(in) :: n
    integer, value :: x(n)
  end subroutine
  subroutine byval0(x)
    integer, value :: x(0)
  end subroutine
  subroutine byval22(x)
    type(tt), value :: x(2, 2)
  end subroutine
  subroutine byval_two(x, y)
    integer, value :: x(3), y(4)
  end subroutine
  subroutine byval_deep(x)
    type(td), value :: x(2)
  end subroutine
  subroutine byval_class1(x)
    class(tt), value :: x(1)
  end subroutine
  subroutine byval_classn(n, x)
    integer, intent(in) :: n
    class(tt), value :: x(n)
  end subroutine
  subroutine byval_parent2(x)
    class(tt), value :: x(2)
  end subroutine
  subroutine byval_star3(x)
    class(*), value :: x(3)
  end subroutine
  subroutine byval_opt_star(n, x)
    integer, intent(in) :: n
    class(*), value, optional :: x(2, n)
  end subroutine
end module

! Static dummy shape: the sequence view has the dummy's shape.
! CHECK-LABEL: func.func @_QPvalue_seq_static
! CHECK: %[[SELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[SSEQ:.*]] = fir.convert %[[SELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: %[[SVIEW:.*]]:2 = hlfir.declare %[[SSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[SCOPY:.*]] = hlfir.as_expr %[[SVIEW]]#0 : (!fir.ref<!fir.array<3xi32>>) -> !hlfir.expr<3xi32>
! CHECK: %[[STMP:.*]]:3 = hlfir.associate %[[SCOPY]](%{{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmPbyval3(%[[STMP]]#0)
! CHECK: hlfir.end_associate %[[STMP]]#1, %[[STMP]]#2
subroutine value_seq_static()
  use m
  integer :: v(4)
  v = [1, 2, 3, 4]
  call byval3(v(2))
end subroutine

! Runtime-shaped dummy: the copied view covers the remaining sequence of the
! base array. The full length computation is bound: length = base size
! minus the column-major offset of the element.
! CHECK-LABEL: func.func @_QPvalue_seq_dynamic
! CHECK: %[[DEXT:.*]] = arith.constant 4 : index
! CHECK: %[[DBSHP:.*]] = fir.shape %[[DEXT]] : (index) -> !fir.shape<1>
! CHECK: %[[DV:.*]]:2 = hlfir.declare %{{.*}}(%[[DBSHP]]) {uniq_name = "_QFvalue_seq_dynamicEv"}
! CHECK: %[[DELT:.*]] = hlfir.designate %[[DV]]#0 (%[[DIDX:.*]])  : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[DSUB:.*]] = arith.subi %[[DIDX]], %[[DONE:.*]] : index
! CHECK: %[[DMUL:.*]] = arith.muli %[[DSUB]], %[[DSTRIDE:.*]] : index
! CHECK: %[[DOFF:.*]] = arith.addi %[[DZERO:.*]], %[[DMUL]] : index
! CHECK: %[[DUBM1:.*]] = arith.subi %[[DEXT]], %[[DONE]] : index
! CHECK: %[[DUB:.*]] = arith.addi %[[DUBM1]], %[[DSTRIDE]] : index
! CHECK: %[[DTOT:.*]] = arith.muli %[[DSTRIDE]], %[[DUB]] : index
! CHECK: %[[DLEN:.*]] = arith.subi %[[DTOT]], %[[DOFF]] : index
! CHECK: %[[DSEQ:.*]] = fir.convert %[[DELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[DSHAPE:.*]] = fir.shape %[[DLEN]] : (index) -> !fir.shape<1>
! CHECK: %[[DVIEW:.*]]:2 = hlfir.declare %[[DSEQ]](%[[DSHAPE]]) {uniq_name = ".sequence.assoc"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
! CHECK: %[[DCOPY:.*]] = hlfir.as_expr %[[DVIEW]]#0 : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
! CHECK: %[[DTMP:.*]]:3 = hlfir.associate %[[DCOPY]](%[[DSHAPE]]) {adapt.valuebyref}
! CHECK: fir.call @_QMmPbyvaln(%{{.*}}, %[[DTMP]]#1)
subroutine value_seq_dynamic()
  use m
  integer :: v(4)
  v = [1, 2, 3, 4]
  call byvaln(3, v(2))
end subroutine

! Zero-extent dummy boundary.
! CHECK-LABEL: func.func @_QPvalue_seq_zero
! CHECK: fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<!fir.array<0xi32>>
! CHECK: hlfir.declare %{{.*}} {uniq_name = ".sequence.assoc"}
! CHECK: fir.call @_QMmPbyval0
subroutine value_seq_zero()
  use m
  integer :: v(4)
  v = [1, 2, 3, 4]
  call byval0(v(4))
end subroutine

! Rank-two dummy: multi-dimensional static view of the sequence.
! CHECK-LABEL: func.func @_QPvalue_seq_rank2
! CHECK: %[[RSEQ:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.type<_QMmTtt{id:i32}>>) -> !fir.ref<!fir.array<2x2x!fir.type<_QMmTtt{id:i32}>>>
! CHECK: %[[RVIEW:.*]]:2 = hlfir.declare %[[RSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[RCOPY:.*]] = hlfir.as_expr %[[RVIEW]]#0
! CHECK: %[[RTMP:.*]]:3 = hlfir.associate %[[RCOPY]]
! CHECK: fir.call @_QMmPbyval22(%[[RTMP]]#0)
subroutine value_seq_rank2()
  use m
  type(tt) :: w(5)
  call byval22(w(2))
end subroutine

! Overlapping element actuals in one call each get their own independent
! sequence-sized temporary.
! CHECK-LABEL: func.func @_QPvalue_seq_overlap
! CHECK: %[[OSEQ1:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: hlfir.declare %[[OSEQ1]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[OTMP1:.*]]:3 = hlfir.associate
! CHECK: %[[OSEQ2:.*]] = fir.convert %{{.*}} : (!fir.ref<i32>) -> !fir.ref<!fir.array<4xi32>>
! CHECK: hlfir.declare %[[OSEQ2]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[OTMP2:.*]]:3 = hlfir.associate
! CHECK: fir.call @_QMmPbyval_two(%[[OTMP1]]#0, %[[OTMP2]]#0)
subroutine value_seq_overlap()
  use m
  integer :: v(4)
  v = [1, 2, 3, 4]
  call byval_two(v(2), v(1))
end subroutine

! Derived type with an allocatable component: the sequence view covers both
! elements so the VALUE copy deep-copies both elements' components.
! CHECK-LABEL: func.func @_QPvalue_seq_deep
! CHECK: %[[PSEQ:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.type<_QMmTtd{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.array<2x!fir.type<_QMmTtd{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
! CHECK: %[[PVIEW:.*]]:2 = hlfir.declare %[[PSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[PCOPY:.*]] = hlfir.as_expr %[[PVIEW]]#0
! CHECK: %[[PTMP:.*]]:3 = hlfir.associate %[[PCOPY]]
! CHECK: fir.call @_QMmPbyval_deep(%[[PTMP]]#0)
subroutine value_seq_deep()
  use m
  type(td) :: b(3)
  call byval_deep(b(2))
end subroutine

! Element of an array component: the remaining sequence is confined to the
! component (its extent is computed from the component shape, including a
! nondefault lower bound).
! CHECK-LABEL: func.func @_QPvalue_seq_component
! CHECK: %[[CMSS:.*]] = fir.shape_shift %[[CMLB:.*]], %[[CMEXT:.*]] : (index, index) -> !fir.shapeshift<1>
! CHECK: %[[CMELT:.*]] = hlfir.designate %{{.*}}{"a"} <%[[CMSS]]> (%[[CMIDX:.*]])  : (!fir.ref<!fir.type<_QMmTtc{a:!fir.array<4xi32>}>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK: %[[CMADD:.*]] = arith.addi %[[CMLB]], %[[CMEXT]] : index
! CHECK: %[[CMUB:.*]] = arith.subi %[[CMADD]], %[[CMONE:.*]] : index
! CHECK: %[[CMSUB:.*]] = arith.subi %[[CMIDX]], %[[CMLB]] : index
! CHECK: %[[CMMUL:.*]] = arith.muli %[[CMSUB]], %[[CMSTRIDE:.*]] : index
! CHECK: %[[CMOFF:.*]] = arith.addi %[[CMZERO:.*]], %[[CMMUL]] : index
! CHECK: %[[CME:.*]] = arith.subi %[[CMUB]], %[[CMLB]] : index
! CHECK: %[[CMEP:.*]] = arith.addi %[[CME]], %[[CMSTRIDE]] : index
! CHECK: %[[CMTOT:.*]] = arith.muli %[[CMSTRIDE]], %[[CMEP]] : index
! CHECK: %[[CMLEN:.*]] = arith.subi %[[CMTOT]], %[[CMOFF]] : index
! CHECK: %[[CMSEQ:.*]] = fir.convert %[[CMELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[CMSHAPE:.*]] = fir.shape %[[CMLEN]] : (index) -> !fir.shape<1>
! CHECK: %[[CMVIEW:.*]]:2 = hlfir.declare %[[CMSEQ]](%[[CMSHAPE]]) {uniq_name = ".sequence.assoc"}
! CHECK: fir.call @_QMmPbyvaln(%{{.*}}, %{{.*}}#1)
subroutine value_seq_component()
  use m
  type(tc) :: vc(2)
  call byvaln(3, vc(2)%a(0))
end subroutine

! Element of a nested rank-two array component with nondefault bounds: the
! remaining extent is computed from the component's own shape.
! CHECK-LABEL: func.func @_QPvalue_seq_nested_rank2
! CHECK: %[[NSS:.*]] = fir.shape_shift %[[NLB1:.*]], %[[NEXT1:.*]], %[[NLB2:.*]], %[[NEXT2:.*]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK: %[[NELT:.*]] = hlfir.designate %{{.*}}{"a"} <%[[NSS]]> (%[[NIDX1:.*]], %[[NIDX2:.*]])  : (!fir.ref<!fir.type<_QMmTinner{a:!fir.array<2x3xi32>}>>, !fir.shapeshift<2>, index, index) -> !fir.ref<i32>
! CHECK: %[[NADD1:.*]] = arith.addi %[[NLB1]], %[[NEXT1]] : index
! CHECK: %[[NUB1:.*]] = arith.subi %[[NADD1]], %[[NONE:.*]] : index
! CHECK: %[[NADD2:.*]] = arith.addi %[[NLB2]], %[[NEXT2]] : index
! CHECK: %[[NUB2:.*]] = arith.subi %[[NADD2]], %[[NONE]] : index
! CHECK: %[[NSUB1:.*]] = arith.subi %[[NIDX1]], %[[NLB1]] : index
! CHECK: %[[NMUL1:.*]] = arith.muli %[[NSUB1]], %[[NSTRIDE:.*]] : index
! CHECK: %[[NOFF1:.*]] = arith.addi %[[NZERO:.*]], %[[NMUL1]] : index
! CHECK: %[[NE1:.*]] = arith.subi %[[NUB1]], %[[NLB1]] : index
! CHECK: %[[NE1P:.*]] = arith.addi %[[NE1]], %[[NSTRIDE]] : index
! CHECK: %[[NSTRIDE2:.*]] = arith.muli %[[NSTRIDE]], %[[NE1P]] : index
! CHECK: %[[NSUB2:.*]] = arith.subi %[[NIDX2]], %[[NLB2]] : index
! CHECK: %[[NMUL2:.*]] = arith.muli %[[NSUB2]], %[[NSTRIDE2]] : index
! CHECK: %[[NOFF:.*]] = arith.addi %[[NOFF1]], %[[NMUL2]] : index
! CHECK: %[[NE2:.*]] = arith.subi %[[NUB2]], %[[NLB2]] : index
! CHECK: %[[NE2P:.*]] = arith.addi %[[NE2]], %[[NSTRIDE]] : index
! CHECK: %[[NTOT:.*]] = arith.muli %[[NSTRIDE2]], %[[NE2P]] : index
! CHECK: %[[NLEN:.*]] = arith.subi %[[NTOT]], %[[NOFF]] : index
! CHECK: %[[NSEQ:.*]] = fir.convert %[[NELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[NSHAPE:.*]] = fir.shape %[[NLEN]] : (index) -> !fir.shape<1>
! CHECK: %[[NVIEW:.*]]:2 = hlfir.declare %[[NSEQ]](%[[NSHAPE]]) {uniq_name = ".sequence.assoc"}
! CHECK: fir.call @_QMmPbyvaln(%{{.*}}, %{{.*}}#1)
subroutine value_seq_nested_rank2()
  use m
  type(outer) :: vo
  call byvaln(3, vo%item(2)%a(0,5))
end subroutine

! Polymorphic dummy: the sequence view has the dummy's static shape and the
! copy is packaged with an array descriptor of matching rank (a
! rank-changing type cast would corrupt the descriptor addendum), used as
! source_box by the sequence-association remapping.
! CHECK-LABEL: func.func @_QPvalue_seq_class
! CHECK: %[[KELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<3x!fir.type<_QMmTtt{id:i32}>>>, index) -> !fir.ref<!fir.type<_QMmTtt{id:i32}>>
! CHECK: %[[KSEQ:.*]] = fir.convert %[[KELT]] : (!fir.ref<!fir.type<_QMmTtt{id:i32}>>) -> !fir.ref<!fir.array<1x!fir.type<_QMmTtt{id:i32}>>>
! CHECK: %[[KVIEW:.*]]:2 = hlfir.declare %[[KSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[KCOPY:.*]] = hlfir.as_expr %[[KVIEW]]#0
! CHECK: %[[KTMP:.*]]:3 = hlfir.associate %[[KCOPY]]
! CHECK: %[[KBOX:.*]] = fir.embox %[[KTMP]]#0(%{{.*}}) : {{.*}} -> !fir.box<!fir.array<1x!fir.type<_QMmTtt{id:i32}>>>
! CHECK: %[[KCLASS:.*]] = fir.convert %[[KBOX]] : (!fir.box<!fir.array<1x!fir.type<_QMmTtt{id:i32}>>>) -> !fir.class<!fir.array<1x!fir.type<_QMmTtt{id:i32}>>>
! CHECK: fir.embox %{{.*}}(%{{.*}}) source_box %[[KCLASS]]
! CHECK: fir.call @_QMmPbyval_class1(%{{.*}}) {{.*}} : (!fir.class<!fir.array<1x!fir.type<_QMmTtt{id:i32}>>>) -> ()
subroutine value_seq_class()
  use m
  type(tt) :: v(3)
  call byval_class1(v(2))
end subroutine

! TYPE(child) element to CLASS(parent) dummy: the view and copy use the
! actual argument element type so the dynamic type and element size are
! preserved; the descriptor uses the dummy declared type over that copy.
! CHECK-LABEL: func.func @_QPvalue_seq_extension
! CHECK: %[[XELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : {{.*}} -> !fir.ref<!fir.type<_QMmTttx{tt:!fir.type<_QMmTtt{id:i32}>,extra:i32}>>
! CHECK: %[[XSEQ:.*]] = fir.convert %[[XELT]] : {{.*}} -> !fir.ref<!fir.array<2x!fir.type<_QMmTttx{tt:!fir.type<_QMmTtt{id:i32}>,extra:i32}>>>
! CHECK: %[[XVIEW:.*]]:2 = hlfir.declare %[[XSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[XCOPY:.*]] = hlfir.as_expr %[[XVIEW]]#0
! CHECK: %[[XTMP:.*]]:3 = hlfir.associate %[[XCOPY]]
! CHECK: %[[XBOX:.*]] = fir.embox %[[XTMP]]#0(%{{.*}}) : {{.*}} -> !fir.box<!fir.array<2x!fir.type<_QMmTttx{tt:!fir.type<_QMmTtt{id:i32}>,extra:i32}>>>
! CHECK: %[[XCLASS:.*]] = fir.convert %[[XBOX]] : {{.*}} -> !fir.class<!fir.array<2x!fir.type<_QMmTtt{id:i32}>>>
! CHECK: fir.embox %{{.*}} source_box %[[XCLASS]]
! CHECK: fir.call @_QMmPbyval_parent2(%{{.*}}) {{.*}} : (!fir.class<!fir.array<2x!fir.type<_QMmTtt{id:i32}>>>) -> ()
subroutine value_seq_extension()
  use m
  type(ttx) :: vx(3)
  call byval_parent2(vx(2))
end subroutine

! INTEGER element to CLASS(*) dummy: the copy keeps the intrinsic element
! type; the rebox adds the addendum the unlimited polymorphic dummy needs.
! CHECK-LABEL: func.func @_QPvalue_seq_unlimited
! CHECK: %[[UELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[USEQ:.*]] = fir.convert %[[UELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: %[[UVIEW:.*]]:2 = hlfir.declare %[[USEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[UCOPY:.*]] = hlfir.as_expr %[[UVIEW]]#0
! CHECK: %[[UTMP:.*]]:3 = hlfir.associate %[[UCOPY]]
! CHECK: %[[UBOX:.*]] = fir.embox %[[UTMP]]#0(%{{.*}}) : {{.*}} -> !fir.box<!fir.array<3xi32>>
! CHECK: %[[UCLASS:.*]] = fir.rebox %[[UBOX]] : (!fir.box<!fir.array<3xi32>>) -> !fir.class<!fir.array<3xnone>>
! CHECK: fir.call @_QMmPbyval_star3(%{{.*}}) {{.*}} : (!fir.class<!fir.array<3xnone>>) -> ()
subroutine value_seq_unlimited()
  use m
  integer :: v(4)
  v = [1, 2, 3, 4]
  call byval_star3(v(2))
end subroutine

! REAL element to an OPTIONAL CLASS(*) rank-two runtime dummy (present
! actual): the view keeps the intrinsic element type and the rebox adds
! the addendum.
! CHECK-LABEL: func.func @_QPvalue_seq_optional_star
! CHECK: %[[OEXT:.*]] = arith.constant 6 : index
! CHECK: %[[OBSHP:.*]] = fir.shape %[[OEXT]] : (index) -> !fir.shape<1>
! CHECK: %[[OV:.*]]:2 = hlfir.declare %{{.*}}(%[[OBSHP]]) {uniq_name = "_QFvalue_seq_optional_starEvr"}
! CHECK: %[[OELT:.*]] = hlfir.designate %[[OV]]#0 (%[[OIDX:.*]])  : (!fir.ref<!fir.array<6xf64>>, index) -> !fir.ref<f64>
! CHECK: %[[OSUB:.*]] = arith.subi %[[OIDX]], %[[OONE:.*]] : index
! CHECK: %[[OMUL:.*]] = arith.muli %[[OSUB]], %[[OSTRIDE:.*]] : index
! CHECK: %[[OOFF:.*]] = arith.addi %[[OZERO:.*]], %[[OMUL]] : index
! CHECK: %[[OUBM1:.*]] = arith.subi %[[OEXT]], %[[OONE]] : index
! CHECK: %[[OUB:.*]] = arith.addi %[[OUBM1]], %[[OSTRIDE]] : index
! CHECK: %[[OTOT:.*]] = arith.muli %[[OSTRIDE]], %[[OUB]] : index
! CHECK: %[[OLEN:.*]] = arith.subi %[[OTOT]], %[[OOFF]] : index
! CHECK: %[[OSEQ:.*]] = fir.convert %[[OELT]] : (!fir.ref<f64>) -> !fir.ref<!fir.array<?xf64>>
! CHECK: %[[OSHAPE:.*]] = fir.shape %[[OLEN]] : (index) -> !fir.shape<1>
! CHECK: %[[OVIEW:.*]]:2 = hlfir.declare %[[OSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: %[[OTMP:.*]]:3 = hlfir.associate
! CHECK: fir.rebox %[[OTMP]]#0 : (!fir.box<!fir.array<?xf64>>) -> !fir.class<!fir.array<?xnone>>
! CHECK: fir.call @_QMmPbyval_opt_star
subroutine value_seq_optional_star()
  use m
  real(8) :: vr(6)
  call byval_opt_star(2, vr(2))
end subroutine

! An element of an assumed-size array passed to a runtime-shaped VALUE dummy:
! the remaining sequence length is unknown, so no sequence view is created
! (the unknown extent must not become a copy length).
! NOTE: the scalar-copy fallback checked here preserves the preexisting
! behavior, which is only correct when the dummy has at most one element; a
! complete implementation must obtain the required dummy extents at a
! suitable argument preparation stage.
! CHECK-LABEL: func.func @_QPvalue_seq_assumed_size
! CHECK: %[[AELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK-NOT: ".sequence.assoc"
! CHECK: %[[ACOPY:.*]] = hlfir.as_expr %[[AELT]] : (!fir.ref<i32>) -> !hlfir.expr<i32>
! CHECK-NOT: ".sequence.assoc"
! CHECK: fir.call @_QMmPbyvaln
subroutine value_seq_assumed_size(a, n)
  use m
  integer :: a(*)
  integer :: n
  call byvaln(n, a(2))
end subroutine

! Positive guard control: an assumed-size base array does not block the
! sequence view when the dummy extents are static.
! CHECK-LABEL: func.func @_QPvalue_seq_assumed_size_static
! CHECK: %[[GELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[GSEQ:.*]] = fir.convert %[[GELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<3xi32>>
! CHECK: %[[GVIEW:.*]]:2 = hlfir.declare %[[GSEQ]](%{{.*}}) {uniq_name = ".sequence.assoc"}
! CHECK: fir.call @_QMmPbyval3
subroutine value_seq_assumed_size_static(a)
  use m
  integer :: a(*)
  call byval3(a(2))
end subroutine

! An element of an assumed-size TYPE array to a runtime-shaped CLASS dummy:
! no sequence view (unknown remaining length), and the scalar copy is
! packaged with a scalar descriptor - rank-consistent, so the descriptor
! addendum stays sound for the one-element case.
! CHECK-LABEL: func.func @_QPvalue_seq_assumed_size_class
! CHECK: %[[SCELT:.*]] = hlfir.designate %{{.*}} (%{{.*}})  : (!fir.box<!fir.array<?x!fir.type<_QMmTtt{id:i32}>>>, index) -> !fir.ref<!fir.type<_QMmTtt{id:i32}>>
! CHECK-NOT: ".sequence.assoc"
! CHECK: %[[SCCOPY:.*]] = hlfir.as_expr %[[SCELT]] : (!fir.ref<!fir.type<_QMmTtt{id:i32}>>) -> !hlfir.expr<!fir.type<_QMmTtt{id:i32}>>
! CHECK: %[[SCTMP:.*]]:3 = hlfir.associate %[[SCCOPY]]
! CHECK: %[[SCBOX:.*]] = fir.embox %[[SCTMP]]#0 : (!fir.ref<!fir.type<_QMmTtt{id:i32}>>) -> !fir.box<!fir.type<_QMmTtt{id:i32}>>
! CHECK: %[[SCCLASS:.*]] = fir.convert %[[SCBOX]] : (!fir.box<!fir.type<_QMmTtt{id:i32}>>) -> !fir.class<!fir.type<_QMmTtt{id:i32}>>
! CHECK: fir.embox %{{.*}}(%{{.*}}) source_box %[[SCCLASS]]
! CHECK: fir.call @_QMmPbyval_classn
subroutine value_seq_assumed_size_class(a, n)
  use m
  type(tt) :: a(*)
  integer :: n
  call byval_classn(n, a(2))
end subroutine
