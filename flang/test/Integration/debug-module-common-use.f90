! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Module COMMON members have no individual fir.global operations. Verify that
! an unqualified USE still makes all of them visible in the using procedure
! when full debug information is emitted, even if executable code does not
! reference them by name.

module testmod
  real :: x0, y0, z0
  common /silly/ x0, y0, z0
contains
  real function dist(x)
    real :: x(3)
    dist = sqrt((x(1) - x0)**2 + (x(2) - y0)**2 + (x(3) - z0)**2)
  end function
  subroutine unused()
  end subroutine
end module

subroutine renamed_common()
  use testmod, local_y => y0
end subroutine

subroutine only_common()
  use testmod, only: local_z => z0
end subroutine

module ancestor_mod
  real :: ma
  common /ancestor_blk/ ma
  interface
    module subroutine separate()
    end subroutine
  end interface
end module

submodule (ancestor_mod) sub_mod
  real :: sa
  common /sub_blk/ sa
contains
  module subroutine separate()
  end subroutine
end submodule

submodule (ancestor_mod:sub_mod) nested_mod
  real :: na
  common /nested_blk/ na
contains
  subroutine nested()
  end subroutine
end submodule

program test
  use testmod
  real :: origo(3), x(3)
  common /silly/ origo
  x = [1.0, 2.0, 3.0]
  print *, dist(x)
end program

! CHECK-DAG: ![[REAL:[0-9]+]] = !DIBasicType(name: "real(kind=4)", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[MAIN:[0-9]+]] = distinct !DISubprogram(name: "test", linkageName: "_QQmain"
! CHECK-DAG: ![[COMMON:[0-9]+]] = !DICommonBlock(scope: ![[MAIN]], declaration: null, name: "silly"
! CHECK-DAG: ![[X0:[0-9]+]] = distinct !DIGlobalVariable(name: "x0", linkageName: "_QMtestmodEx0", scope: ![[COMMON]],{{.*}}type: ![[REAL]]
! CHECK-DAG: ![[Y0:[0-9]+]] = distinct !DIGlobalVariable(name: "y0", linkageName: "_QMtestmodEy0", scope: ![[COMMON]],{{.*}}type: ![[REAL]]
! CHECK-DAG: ![[Z0:[0-9]+]] = distinct !DIGlobalVariable(name: "z0", linkageName: "_QMtestmodEz0", scope: ![[COMMON]],{{.*}}type: ![[REAL]]
! CHECK-DAG: ![[X0_EXPR:[0-9]+]] = !DIGlobalVariableExpression(var: ![[X0]], expr: !DIExpression())
! CHECK-DAG: ![[Y0_EXPR:[0-9]+]] = !DIGlobalVariableExpression(var: ![[Y0]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[Z0_EXPR:[0-9]+]] = !DIGlobalVariableExpression(var: ![[Z0]], expr: !DIExpression(DW_OP_plus_uconst, 8))
! CHECK-DAG: @silly_ = common global {{.*}}!dbg ![[X0_EXPR]]{{.*}}!dbg ![[Y0_EXPR]]{{.*}}!dbg ![[Z0_EXPR]]

! The name in the using scope is preserved for renames, both with and without
! ONLY.
! CHECK-DAG: ![[RENAMED:[0-9]+]] = distinct !DISubprogram(name: "renamed_common", linkageName: "renamed_common_"
! CHECK-DAG: ![[RENAMED_BLOCK:[0-9]+]] = !DICommonBlock(scope: ![[RENAMED]], declaration: null, name: "silly"
! CHECK-DAG: distinct !DIGlobalVariable(name: "local_y", linkageName: "_QMtestmodEy0", scope: ![[RENAMED_BLOCK]], file: !{{[0-9]+}}, line: 9,
! CHECK-DAG: ![[ONLY:[0-9]+]] = distinct !DISubprogram(name: "only_common", linkageName: "only_common_"
! CHECK-DAG: ![[ONLY_BLOCK:[0-9]+]] = !DICommonBlock(scope: ![[ONLY]], declaration: null, name: "silly"
! CHECK-DAG: distinct !DIGlobalVariable(name: "local_z", linkageName: "_QMtestmodEz0", scope: ![[ONLY_BLOCK]]

! Host-associated COMMON members use the same compact metadata from
! fir.module_debug_imports. The block is located at its COMMON statement, and
! each member at its declaration.
! CHECK-DAG: ![[UNUSED:[0-9]+]] = distinct !DISubprogram(name: "unused", linkageName: "_QMtestmodPunused"
! CHECK-DAG: ![[HOST_BLOCK:[0-9]+]] = !DICommonBlock(scope: ![[UNUSED]], declaration: null, name: "silly", file: !{{[0-9]+}}, line: 10)
! CHECK-DAG: distinct !DIGlobalVariable(name: "x0", linkageName: "_QMtestmodEx0", scope: ![[HOST_BLOCK]], file: !{{[0-9]+}}, line: 9,
! CHECK-DAG: distinct !DIGlobalVariable(name: "y0", linkageName: "_QMtestmodEy0", scope: ![[HOST_BLOCK]]
! CHECK-DAG: distinct !DIGlobalVariable(name: "z0", linkageName: "_QMtestmodEz0", scope: ![[HOST_BLOCK]]

! A procedure of a submodule sees the COMMON members of the submodule and of
! its whole ancestry.
! CHECK-DAG: ![[SEPARATE:[0-9]+]] = distinct !DISubprogram(name: "separate", linkageName: "_QMancestor_modPseparate"
! CHECK-DAG: ![[SEPARATE_ANCESTOR:[0-9]+]] = !DICommonBlock(scope: ![[SEPARATE]], declaration: null, name: "ancestor_blk"
! CHECK-DAG: distinct !DIGlobalVariable(name: "ma", linkageName: "_QMancestor_modEma", scope: ![[SEPARATE_ANCESTOR]]
! CHECK-DAG: ![[SEPARATE_SUB:[0-9]+]] = !DICommonBlock(scope: ![[SEPARATE]], declaration: null, name: "sub_blk"
! CHECK-DAG: distinct !DIGlobalVariable(name: "sa", linkageName: "_QMancestor_modSsub_modEsa", scope: ![[SEPARATE_SUB]]
! CHECK-DAG: ![[NESTED:[0-9]+]] = distinct !DISubprogram(name: "nested", linkageName: "_QMancestor_modSsub_modSnested_modPnested"
! CHECK-DAG: ![[NESTED_ANCESTOR:[0-9]+]] = !DICommonBlock(scope: ![[NESTED]], declaration: null, name: "ancestor_blk"
! CHECK-DAG: distinct !DIGlobalVariable(name: "ma", linkageName: "_QMancestor_modEma", scope: ![[NESTED_ANCESTOR]]
! CHECK-DAG: ![[NESTED_SUB:[0-9]+]] = !DICommonBlock(scope: ![[NESTED]], declaration: null, name: "sub_blk"
! CHECK-DAG: distinct !DIGlobalVariable(name: "sa", linkageName: "_QMancestor_modSsub_modEsa", scope: ![[NESTED_SUB]]
! CHECK-DAG: ![[NESTED_OWN:[0-9]+]] = !DICommonBlock(scope: ![[NESTED]], declaration: null, name: "nested_blk"
! CHECK-DAG: distinct !DIGlobalVariable(name: "na", linkageName: "_QMancestor_modSsub_modSnested_modEna", scope: ![[NESTED_OWN]]
