! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

subroutine f1
  real(kind=4) :: x, y, xa, ya
  common // x, y
  common /a/ xa, ya
  x = 1.1
  y = 2.2
  xa = 3.3
  ya = 4.4
  print *, x, y, xa, ya
end subroutine
! CHECK-DAG: ![[XF1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "x", linkageName: "_QFf1Ex", scope: ![[CBF1:[0-9]+]], file: !5, line: [[@LINE-9]], type: ![[REAL:[0-9]+]]{{.*}})
! CHECK-DAG: ![[EXPXF1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XF1]], expr: !DIExpression())
! CHECK-DAG: ![[YF1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "y", linkageName: "_QFf1Ey", scope: ![[CBF1]], file: !{{[0-9]+}}, line: [[@LINE-11]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPYF1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[YF1]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[XAF1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "xa", linkageName: "_QFf1Exa", scope: ![[CBAF1:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-13]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPXAF1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XAF1]], expr: !DIExpression())
! CHECK-DAG: ![[YAF1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "ya", linkageName: "_QFf1Eya", scope: ![[CBAF1]], file: !{{[0-9]+}}, line: [[@LINE-15]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPYAF1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[YAF1]], expr: !DIExpression(DW_OP_plus_uconst, 4))


subroutine f2
  real(kind=4) :: x, y, z, xa, ya, za
  common // x, y, z
  common /a/ xa, ya, za
  print *, x, y, z, xa, ya, za
end subroutine
! CHECK-DAG: ![[XF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "x", linkageName: "_QFf2Ex", scope: ![[CBF2:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-5]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPXF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XF2]], expr: !DIExpression())
! CHECK-DAG: ![[YF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "y", linkageName: "_QFf2Ey", scope: ![[CBF2]], file: !{{[0-9]+}}, line: [[@LINE-7]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPYF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[YF2]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[ZF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "z", linkageName: "_QFf2Ez", scope: ![[CBF2]], file: !{{[0-9]+}}, line: [[@LINE-9]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPZF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[ZF2]], expr: !DIExpression(DW_OP_plus_uconst, 8))
! CHECK-DAG: ![[XAF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "xa", linkageName: "_QFf2Exa", scope: ![[CBAF2:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-11]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPXAF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XAF2]], expr: !DIExpression())
! CHECK-DAG: ![[YAF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "ya", linkageName: "_QFf2Eya", scope: ![[CBAF2]], file: !{{[0-9]+}}, line: [[@LINE-13]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPYAF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[YAF2]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[ZAF2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "za", linkageName: "_QFf2Eza", scope: ![[CBAF2]], file: !{{[0-9]+}}, line: [[@LINE-15]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPZAF2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[ZAF2]], expr: !DIExpression(DW_OP_plus_uconst, 8))

subroutine f3
  integer(kind=4) :: x = 42, xa = 42
  common // x
  common /a/ xa
  print *, x
  print *, xa
end subroutine
! CHECK-DAG: ![[XF3:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "x", linkageName: "_QFf3Ex", scope: ![[CBF3:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-6]], type: ![[INT:[0-9]+]]{{.*}})
! CHECK-DAG: ![[EXPXF3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XF3]], expr: !DIExpression())
! CHECK-DAG: ![[XAF3:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "xa", linkageName: "_QFf3Exa", scope: ![[CBAF3:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-8]], type: ![[INT]]{{.*}})
! CHECK-DAG: ![[EXPXAF3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[XAF3]], expr: !DIExpression())

program test
  real(kind=4) :: v1, v2, v3, va1, va2, va3
  common // v1, v2, v3
  common /a/ va1, va2, va3
  call f1()
  call f2()
  call f3()
  print *, v1, va1, va3
END
! CHECK-DAG: ![[V1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "v1", linkageName: "_QFEv1", scope: ![[CBM:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-8]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPV1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[V1]], expr: !DIExpression())
! CHECK-DAG: ![[V2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "v2", linkageName: "_QFEv2", scope: ![[CBM]], file: !{{[0-9]+}}, line: [[@LINE-10]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPV2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[V2]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[V3:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "v3", linkageName: "_QFEv3", scope: ![[CBM]], file: !{{[0-9]+}}, line: [[@LINE-12]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPV3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[V3]], expr: !DIExpression(DW_OP_plus_uconst, 8))
! CHECK-DAG: ![[VA1:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "va1", linkageName: "_QFEva1", scope: ![[CBAM:[0-9]+]], file: !{{[0-9]+}}, line: [[@LINE-14]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPVA1:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VA1]], expr: !DIExpression())
! CHECK-DAG: ![[VA2:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "va2", linkageName: "_QFEva2", scope: ![[CBAM]], file: !{{[0-9]+}}, line: [[@LINE-16]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPVA2:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VA2]], expr: !DIExpression(DW_OP_plus_uconst, 4))
! CHECK-DAG: ![[VA3:[0-9]+]] = {{.*}}!DIGlobalVariable(name: "va3", linkageName: "_QFEva3", scope: ![[CBAM]], file: !{{[0-9]+}}, line: [[@LINE-18]], type: ![[REAL]]{{.*}})
! CHECK-DAG: ![[EXPVA3:[0-9]+]] = !DIGlobalVariableExpression(var: ![[VA3]], expr: !DIExpression(DW_OP_plus_uconst, 8))


! CHECK-DAG: ![[REAL]] = !DIBasicType(name: "real", size: 32, encoding: DW_ATE_float)
! CHECK-DAG: ![[INT]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)

! CHECK-DAG: ![[F1:[0-9]+]] = {{.*}}!DISubprogram(name: "f1"{{.*}})
! CHECK-DAG: ![[CBF1]] = !DICommonBlock(scope: ![[F1]], declaration: null, name: "__BLNK__"{{.*}})
! CHECK-DAG: ![[CBAF1]] = !DICommonBlock(scope: ![[F1]], declaration: null, name: "a"{{.*}})

! CHECK-DAG: ![[F2:[0-9]+]] = {{.*}}!DISubprogram(name: "f2"{{.*}})
! CHECK-DAG: ![[CBF2]] = !DICommonBlock(scope: ![[F2]], declaration: null, name: "__BLNK__"{{.*}})
! CHECK-DAG: ![[CBAF2]] = !DICommonBlock(scope: ![[F2]], declaration: null, name: "a"{{.*}})

! CHECK-DAG: ![[F3:[0-9]+]] = {{.*}}!DISubprogram(name: "f3"{{.*}})
! CHECK-DAG: ![[CBF3]] = !DICommonBlock(scope: ![[F3]], declaration: null, name: "__BLNK__"{{.*}})
! CHECK-DAG: ![[CBAF3]] = !DICommonBlock(scope: ![[F3]], declaration: null, name: "a"{{.*}})

! CHECK-DAG: ![[MAIN:[0-9]+]] = {{.*}}!DISubprogram(name: "test"{{.*}})
! CHECK-DAG: ![[CBM]] = !DICommonBlock(scope: ![[MAIN]], declaration: null, name: "__BLNK__"{{.*}})
! CHECK-DAG: ![[CBAM]] = !DICommonBlock(scope: ![[MAIN]], declaration: null, name: "a"{{.*}})

! Using CHECK-DAG-SAME so that we are not dependent on order of variable in these lists.
! CHECK-DAG: @__BLNK__ = global{{.*}}
! CHECK-DAG-SAME: !dbg ![[EXPXF1]]
! CHECK-DAG-SAME: !dbg ![[EXPYF1]]
! CHECK-DAG-SAME: !dbg ![[EXPXF2]]
! CHECK-DAG-SAME: !dbg ![[EXPYF2]]
! CHECK-DAG-SAME: !dbg ![[EXPZF2]]
! CHECK-DAG-SAME: !dbg ![[EXPXF3]]
! CHECK-DAG-SAME: !dbg ![[EXPV1]]
! CHECK-DAG-SAME: !dbg ![[EXPV2]]
! CHECK-DAG-SAME: !dbg ![[EXPV3]]

! CHECK-DAG: @a_ = global{{.*}}
! CHECK-DAG-SAME: !dbg ![[EXPXAF1]]
! CHECK-DAG-SAME: !dbg ![[EXPYAF1]]
! CHECK-DAG-SAME: !dbg ![[EXPXAF2]]
! CHECK-DAG-SAME: !dbg ![[EXPYAF2]]
! CHECK-DAG-SAME: !dbg ![[EXPZAF2]]
! CHECK-DAG-SAME: !dbg ![[EXPXAF3]]
! CHECK-DAG-SAME: !dbg ![[EXPVA1]]
! CHECK-DAG-SAME: !dbg ![[EXPVA2]]
! CHECK-DAG-SAME: !dbg ![[EXPVA3]]

! CHECK-DAG: !DICompileUnit({{.*}}, globals: ![[GLOBALS:[0-9]+]])
! CHECK-DAG: ![[GLOBALS]]
! CHECK-DAG-SAME: ![[EXPXF1]]
! CHECK-DAG-SAME: ![[EXPYF1]]
! CHECK-DAG-SAME: ![[EXPXAF1]]
! CHECK-DAG-SAME: ![[EXPYAF1]]
! CHECK-DAG-SAME: ![[EXPXF2]]
! CHECK-DAG-SAME: ![[EXPYF2]]
! CHECK-DAG-SAME: ![[EXPZF2]]
! CHECK-DAG-SAME: ![[EXPXAF2]]
! CHECK-DAG-SAME: ![[EXPYAF2]]
! CHECK-DAG-SAME: ![[EXPZAF2]]
! CHECK-DAG-SAME: ![[EXPXF3]]
! CHECK-DAG-SAME: ![[EXPXAF3]]
! CHECK-DAG-SAME: ![[EXPV1]]
! CHECK-DAG-SAME: ![[EXPV2]]
! CHECK-DAG-SAME: ![[EXPV3]]
! CHECK-DAG-SAME: ![[EXPVA1]]
! CHECK-DAG-SAME: ![[EXPVA2]]
! CHECK-DAG-SAME: ![[EXPVA3]]
