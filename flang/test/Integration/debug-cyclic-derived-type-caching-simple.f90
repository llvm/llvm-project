! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

! Simple test that checks that metadata for `t0` is only duplicated once.

! The difficulty is that at the mlir::LLVM::DITypeAttr, because of the
! lack of MLIR attribute true recursion, the mlir::LLVM::DITypeAttr for
! `t0` inside `t1` is special because it is only valid when found
! in an mlir::LLVM::DITypeAttr tree under `base` (it will point to `base`
! via an integer id that is only meanigfu;l when a node with such id has
! been traversed).
! A different node has to be created for `t0` usage in `x0` (will
! point to the actual mlir::LLVM::DITypeAttr for `base` instead of
! an integer id since the cycle is already "taken care of" in `base`
! definition).
! However, the same special `t0` node can be used for both `x1_1` and
! `x1_2` components because they are both under `base` in the
! mlir::LLVM::DITypeAttr tree definition. This used to not be the case,
! leading to a lot of duplicate mlir::LLVM::DITypeAttr and actual LLVM IR
! metadata, causing noticeable compilation slowdowns in apps with non trivial
! derived types.

subroutine duplicate_cycle_branch()
  type t0
    type(base), pointer :: x
  end type
  type t1
    type(t0) :: x1_1
    type(t0) :: x1_2
  end type
  type base
    type(t1) :: x_base
  end type
  type(base) :: x
  type(t0) :: x0
  call bar(x, x0)
end subroutine
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t0",
! CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t0",
! CHECK-NOT: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t0",
