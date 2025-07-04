! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Test that debug metadata for derived types is not duplicated more than needed
! when emitting debug info with non trivial cycles.


! In the type graph below, G has a back edge to B, and F to D.
! This causes C to be in the middle of B cycle, and E to be
! both in B and D cycles.
! C and E are used in several contexts, under B, under D, and outside
! of it to test how metadata is generated for them.
!
! Without "local caching" of C and E when generating mlir::LLVM::DITypeAttr
! for such derived types, many duplicate llvm metadata for the derived types
! would be emitted, while with the right duplication of mlir::LLVM::DITypeAttr,
! a lot more duplicate llvm IR metadata ends up emitted (19 DICompositeType
! vs 71 before the patch that added this test).
!
!
!  A -> B -> C -> D -> E -> F -> G -> B
!  |    |         |         |
!  |    |         |         | -> D
!  |    |         |
!  |    |         | -> H -> E
!  |    |
!  |    | -> I -> E
!  |         | -> C
!  |
!  | -> C
!  | -> E

subroutine type_cycles_caching()
  type g
    type(b), pointer :: c_b
  end type
  type f
    type(g) :: c_b
    type(d), pointer :: c_d
  end type
  type e
    type(f) :: c_f
  end type
    type h
      ! Can reuse metadata of type(e) under 'd'.
      type(e) :: c_e
    end type
  type d
    type(e) :: c_e
    type(h) :: c_h
  end type
  type c
    type(d) :: c_d
  end type
    type i
      ! Cannot reuse metadata of type(e) under 'd'.
      type(e) :: c_e
      ! Can reuse metadata of type(c) under 'b'.
      type(c) :: c_c
    end type
  type b
    type(c) :: c_c
    type(i) :: c_i
  end type
  type a
    type(b) :: c_b
    ! Cannot reuse metadata of type(c) under 'b'
    type(c) :: c_c
    ! Cannot reuse metadata of type(e) under 'd', nor the one under 'b'
    type(e) :: c_e
  end type
  type(a) :: xa
  ! Can reuse metadata of type(c) for xa%c_c
  type(c) :: xc
  ! Can reuse metadata of type(c) for xa%c_e
  type(e) :: xe
  call bar(xa, xc, xe)
end subroutine

! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "a",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "b",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "c",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "d",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "e",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "f",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "g",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "h",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "i",
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "e"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "f"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "c"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "d"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "e"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "f"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "g"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "h"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "e"
! CHECK-NOT: distinct !DICompositeType
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "f"
! CHECK-NOT: distinct !DICompositeType
