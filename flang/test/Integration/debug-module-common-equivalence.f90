! RUN: %flang_fc1 -emit-hlfir -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=HLFIR
! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s --check-prefix=LLVM

! EQUIVALENCE aliases of module COMMON members are not listed in the COMMON
! statement. Full debug info still describes them at their offset in the
! block, including from a USE that never references them.

module eqmod
  integer :: a, c
  integer :: b
  integer :: w(2)
  integer :: v
  common /blk/ a, c
  common /arr/ w
  equivalence (a, b)
  equivalence (w(2), v)
contains
  subroutine unused()
  end subroutine
end module

subroutine use_all()
  use eqmod
end subroutine

subroutine only_alias()
  use eqmod, only: local_b => b, v
end subroutine

program test
  use eqmod
  b = 1
  v = 2
end program

! A contained procedure that does not reference the aliases has no declare for
! them. Their debug entries come from fir.module_debug_imports.
! HLFIR-LABEL: func.func @_QMeqmodPunused()
! HLFIR-NOT: hlfir.declare
! HLFIR: return

! HLFIR-LABEL: func.func @_QPuse_all()
! HLFIR: fir.use_stmt "eqmod" common_block_members
! HLFIR-SAME: #fir.use_common_block_member<"w", "_QMeqmodEw", @arr_, !fir.array<2xi32>, 0, #loc
! HLFIR-SAME: #fir.use_common_block_member<"v", "_QMeqmodEv", @arr_, i32, 4, #loc
! HLFIR-SAME: #fir.use_common_block_member<"a", "_QMeqmodEa", @blk_, i32, 0, #loc
! HLFIR-SAME: #fir.use_common_block_member<"c", "_QMeqmodEc", @blk_, i32, 4, #loc
! HLFIR-SAME: #fir.use_common_block_member<"b", "_QMeqmodEb", @blk_, i32, 0, #loc
! HLFIR: return

! HLFIR-LABEL: func.func @_QPonly_alias()
! HLFIR: fir.use_stmt "eqmod" only_symbols
! HLFIR-SAME: #fir.use_common_block_member<"local_b", "_QMeqmodEb", @blk_, i32, 0, #loc
! HLFIR-NOT: #fir.use_common_block_member<"a"
! HLFIR-NOT: #fir.use_common_block_member<"c"
! HLFIR: return

! HLFIR: fir.module_debug_imports "eqmod" common_block_members
! HLFIR-SAME: #fir.use_common_block_member<"w", "_QMeqmodEw", @arr_, !fir.array<2xi32>, 0, #loc
! HLFIR-SAME: #fir.use_common_block_member<"v", "_QMeqmodEv", @arr_, i32, 4, #loc
! HLFIR-SAME: #fir.use_common_block_member<"a", "_QMeqmodEa", @blk_, i32, 0, #loc
! HLFIR-SAME: #fir.use_common_block_member<"b", "_QMeqmodEb", @blk_, i32, 0, #loc

! LLVM-DAG: ![[INT:.*]] = !DIBasicType(name: "integer(kind=4)", size: 32, encoding: DW_ATE_signed)

! An unqualified USE describes both the COMMON objects and the aliases.
! LLVM-DAG: ![[USE_ALL:.*]] = distinct !DISubprogram(name: "use_all", linkageName: "use_all_"
! LLVM-DAG: ![[BLK:.*]] = !DICommonBlock(scope: ![[USE_ALL]], declaration: null, name: "blk"
! LLVM-DAG: ![[ARR:.*]] = !DICommonBlock(scope: ![[USE_ALL]], declaration: null, name: "arr"
! LLVM-DAG: ![[B:.*]] = distinct !DIGlobalVariable(name: "b", linkageName: "_QMeqmodEb", scope: ![[BLK]],{{.*}}type: ![[INT]]
! LLVM-DAG: !DIGlobalVariableExpression(var: ![[B]], expr: !DIExpression())
! LLVM-DAG: ![[V:.*]] = distinct !DIGlobalVariable(name: "v", linkageName: "_QMeqmodEv", scope: ![[ARR]],{{.*}}type: ![[INT]]
! LLVM-DAG: !DIGlobalVariableExpression(var: ![[V]], expr: !DIExpression(DW_OP_plus_uconst, 4))

! A contained procedure that does not reference the aliases still sees them
! through fir.module_debug_imports.
! LLVM-DAG: ![[UNUSED:.*]] = distinct !DISubprogram(name: "unused", linkageName: "_QMeqmodPunused"
! LLVM-DAG: ![[HOST_BLK:.*]] = !DICommonBlock(scope: ![[UNUSED]], declaration: null, name: "blk"
! LLVM-DAG: ![[HOST_ARR:.*]] = !DICommonBlock(scope: ![[UNUSED]], declaration: null, name: "arr"
! LLVM-DAG: ![[HOST_B:.*]] = distinct !DIGlobalVariable(name: "b", linkageName: "_QMeqmodEb", scope: ![[HOST_BLK]],{{.*}}type: ![[INT]]
! LLVM-DAG: !DIGlobalVariableExpression(var: ![[HOST_B]], expr: !DIExpression())
! LLVM-DAG: ![[HOST_V:.*]] = distinct !DIGlobalVariable(name: "v", linkageName: "_QMeqmodEv", scope: ![[HOST_ARR]],{{.*}}type: ![[INT]]
! LLVM-DAG: !DIGlobalVariableExpression(var: ![[HOST_V]], expr: !DIExpression(DW_OP_plus_uconst, 4))

! ONLY keeps the local name of the alias and does not import the other block.
! LLVM-DAG: ![[ONLY:.*]] = distinct !DISubprogram(name: "only_alias", linkageName: "only_alias_"
! LLVM-DAG: ![[ONLY_BLK:.*]] = !DICommonBlock(scope: ![[ONLY]], declaration: null, name: "blk"
! LLVM-DAG: distinct !DIGlobalVariable(name: "local_b", linkageName: "_QMeqmodEb", scope: ![[ONLY_BLK]]
