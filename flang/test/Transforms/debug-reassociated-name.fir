// RUN: split-file %s %t
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/reassoc.fir | FileCheck %t/reassoc.fir
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/nest.fir | FileCheck %t/nest.fir
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/inc.fir | FileCheck %t/inc.fir
// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %t/inl.fir | FileCheck %t/inl.fir

// Test the scopes of a name declared in more than one place. Each part holds a
// procedure, its locations and the checks of its output.

//--- reassoc.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPreassoc(%arg0: !fir.ref<i32>) {
    %true = arith.constant true loc(#func)
    %ds = fir.undefined !fir.dscope loc(#func)
    %0 = fircg.ext_declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFreassocEsel") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#decl_sel)
    // Declared once.
    %1 = fircg.ext_declare %arg0 uniq_name("_QFreassocEonce") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_once)
    cf.cond_br %true, ^bb1, ^bb2 loc(#sel)
  ^bb1:
    %2 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#branch1)
    %3 = fircg.ext_declare %2 uniq_name("_QFreassocEsel") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_sel1)
    %4 = fir.load %3 : !fir.ref<i32> loc(#branch1)
    // The other branch returns, so the code after the construct is in the
    // block of this branch.
    %5 = fir.load %0 : !fir.ref<i32> loc(#after)
    return loc(#ret)
  ^bb2:
    %6 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#branch2)
    %7 = fircg.ext_declare %6 uniq_name("_QFreassocEsel") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_sel2)
    return loc(#branch2)
  } loc(#func)
} loc(#mod)

#mod = loc("t.f90":1:1)
#func = loc("t.f90":3:1)
#decl_once = loc("t.f90":4:14)
// All the declarations of "sel" share a location.
#decl_sel = loc("t.f90":5:14)
#sel = loc("t.f90":7:3)
#after = loc("t.f90":10:3)
#ret = loc("t.f90":12:1)
// The code of each branch, here on one line, is tagged with its scope.
#scope1 = #fir.lexical_scope<loc = loc("t.f90":8:5)>
#scope2 = #fir.lexical_scope<loc = loc("t.f90":8:20)>
#branch1 = loc(fused<#scope1>["t.f90":8:5])
#branch2 = loc(fused<#scope2>["t.f90":8:20])
#decl_sel1 = loc(fused<#scope1>[#decl_sel])
#decl_sel2 = loc(fused<#scope2>[#decl_sel])

// CHECK: func.func @_QPreassoc
// CHECK: ^bb1:
// CHECK: fircg.ext_declare {{.*}} loc(#[[SEL1_DECL:loc[0-9]+]])
// CHECK-NEXT: fir.load {{.*}} loc(#[[BRANCH1:loc[0-9]+]])
// CHECK-NEXT: fir.load {{.*}} loc(#[[AFTER:loc[0-9]+]])

// CHECK-DAG: #[[SP:.+]] = #llvm.di_subprogram<{{.*}}name = "reassoc"{{.*}}>

// A declaration outside of the branches stays in the procedure.
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SP]], name = "sel"{{.*}}arg = 1

// Each branch has its own scope, positioned at its statement, even on one line.
// CHECK-DAG: #[[B1:.+]] = #llvm.di_lexical_block<scope = #[[SP]], {{.*}}line = 8, column = 5>
// CHECK-DAG: #[[B2:.+]] = #llvm.di_lexical_block<scope = #[[SP]], {{.*}}line = 8, column = 20>
// CHECK-DAG: #[[SEL1:.+]] = #llvm.di_local_variable<scope = #[[B1]], name = "sel"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[B2]], name = "sel"

// The declaration is in the scope of its branch.
// CHECK-DAG: #[[SEL1_DECL]] = loc(fused<#[[SEL1]]>[#[[SEL1_SCOPED:loc[0-9]+]]])
// CHECK-DAG: #[[SEL1_SCOPED]] = loc(fused<#[[B1]]>[#{{loc[0-9]+}}])

// The scope of a branch covers the code tagged with it, and not the code after
// the construct in the same block.
// CHECK-DAG: #[[BRANCH1]] = loc(fused<#[[B1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[AFTER]] = loc("t.f90":10:3)

// A name declared once stays in the procedure.
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SP]], name = "once"

//--- nest.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPnest(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
    %true = arith.constant true loc(#nest)
    %0 = fircg.ext_declare %arg0 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p)
    %1 = fircg.ext_declare %arg1 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q)
    cf.cond_br %true, ^bb1, ^bb5 loc(#sel_p)
  ^bb1:
    %2 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#p_branch1)
    %3 = fircg.ext_declare %2 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p1)
    cf.cond_br %true, ^bb2, ^bb3 loc(#sel_q)
  ^bb2:
    %4 = fir.convert %arg1 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#q_branch1)
    %5 = fircg.ext_declare %4 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q1)
    cf.br ^bb4 loc(#q_branch1)
  ^bb3:
    %6 = fir.convert %arg1 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#q_branch2)
    %7 = fircg.ext_declare %6 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q2)
    cf.br ^bb4 loc(#q_branch2)
  ^bb4:
    %8 = fir.load %3 : !fir.ref<i32> loc(#loop)
    cf.cond_br %true, ^bb4, ^bb6 loc(#loop)
  ^bb5:
    %9 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#p_branch2)
    %10 = fircg.ext_declare %9 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p2)
    cf.br ^bb6 loc(#p_branch2)
  ^bb6:
    return loc(#nest_ret)
  } loc(#nest)
} loc(#mod)

#mod = loc("t.f90":1:1)
#nest = loc("t.f90":20:1)
#decl_p = loc("t.f90":21:14)
#decl_q = loc("t.f90":21:17)
#sel_p = loc("t.f90":22:3)
#nest_ret = loc("t.f90":35:1)
// The construct for "q" is in the first branch of the one for "p".
#p_scope1 = #fir.lexical_scope<loc = loc("t.f90":23:5)>
#p_scope2 = #fir.lexical_scope<loc = loc("t.f90":33:5)>
#q_scope1 = #fir.lexical_scope<loc = loc("t.f90":25:7), parent = #p_scope1>
#q_scope2 = #fir.lexical_scope<loc = loc("t.f90":27:7), parent = #p_scope1>
#p_branch1 = loc(fused<#p_scope1>["t.f90":23:5])
#sel_q = loc(fused<#p_scope1>["t.f90":24:5])
#q_branch1 = loc(fused<#q_scope1>["t.f90":25:7])
#q_branch2 = loc(fused<#q_scope2>["t.f90":27:7])
#loop = loc(fused<#p_scope1>["t.f90":30:7])
#p_branch2 = loc(fused<#p_scope2>["t.f90":33:5])
#decl_p1 = loc(fused<#p_scope1>[#decl_p])
#decl_p2 = loc(fused<#p_scope2>[#decl_p])
#decl_q1 = loc(fused<#q_scope1>[#decl_q])
#decl_q2 = loc(fused<#q_scope2>[#decl_q])

// CHECK: func.func @_QPnest
// CHECK: ^bb4:
// CHECK-NEXT: fir.load {{.*}} loc(#[[LOOP:loc[0-9]+]])
// CHECK: ^bb6:
// CHECK: return loc(#[[NEST_RET:loc[0-9]+]])

// The scopes of a nested construct are nested in the scope of its branch.
// CHECK-DAG: #[[NEST:.+]] = #llvm.di_subprogram<{{.*}}name = "nest"{{.*}}>
// CHECK-DAG: #[[P1:.+]] = #llvm.di_lexical_block<scope = #[[NEST]], {{.*}}line = 23, column = 5>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[P1]], name = "p"
// CHECK-DAG: #[[Q1:.+]] = #llvm.di_lexical_block<scope = #[[P1]], {{.*}}line = 25, column = 7>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[Q1]], name = "q"
// CHECK-DAG: #[[Q2:.+]] = #llvm.di_lexical_block<scope = #[[P1]], {{.*}}line = 27, column = 7>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[Q2]], name = "q"
// CHECK-DAG: #[[P2:.+]] = #llvm.di_lexical_block<scope = #[[NEST]], {{.*}}line = 33, column = 5>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[P2]], name = "p"

// The scope of a branch covers its blocks and ends with the branch.
// CHECK-DAG: #[[LOOP]] = loc(fused<#[[P1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[NEST_RET]] = loc("t.f90":35:1)

//--- inc.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPinc(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
    %true = arith.constant true loc(#inc)
    %0 = fircg.ext_declare %arg0 uniq_name("_QFincEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_x)
    %1 = fircg.ext_declare %arg1 uniq_name("_QFincEy") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_y)
    cf.cond_br %true, ^bb1, ^bb2 loc(#x_sel)
  ^bb1:
    %2 = fircg.ext_declare %arg0 uniq_name("_QFincEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_x1)
    // Included in the branch.
    %3 = fir.load %2 : !fir.ref<i32> loc(#x_included)
    cf.br ^bb3 loc(#x_branch1)
  ^bb2:
    %4 = fircg.ext_declare %arg0 uniq_name("_QFincEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_x2)
    cf.br ^bb3 loc(#x_branch2)
  ^bb3:
    // A construct in an included file.
    cf.cond_br %true, ^bb4, ^bb5 loc(#y_sel)
  ^bb4:
    %5 = fircg.ext_declare %arg1 uniq_name("_QFincEy") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_y1)
    %6 = fir.load %5 : !fir.ref<i32> loc(#y_branch1)
    cf.br ^bb6 loc(#y_branch1)
  ^bb5:
    %7 = fircg.ext_declare %arg1 uniq_name("_QFincEy") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_y2)
    %8 = fir.load %7 : !fir.ref<i32> loc(#y_branch2)
    cf.br ^bb6 loc(#y_branch2)
  ^bb6:
    %9 = fir.load %0 : !fir.ref<i32> loc(#inc_after)
    return loc(#inc_ret)
  } loc(#inc)
} loc(#mod)

#mod = loc("t.f90":1:1)
#inc = loc("t.f90":40:1)
#decl_x = loc("t.f90":41:14)
#decl_y = loc("t.f90":41:17)
#x_sel = loc("t.f90":42:3)
#x_scope1 = #fir.lexical_scope<loc = loc("t.f90":43:5)>
#x_scope2 = #fir.lexical_scope<loc = loc("t.f90":45:5)>
#x_branch1 = loc(fused<#x_scope1>["t.f90":43:5])
#x_include = loc("t.f90":44:5)
#x_included = loc(fused<#x_scope1>[fused<#fir<loc_kind_array[<base>, <inclusion>]>>["a.inc":1:3, #x_include]])
#x_branch2 = loc(fused<#x_scope2>["t.f90":45:5])
#decl_x1 = loc(fused<#x_scope1>[#decl_x])
#decl_x2 = loc(fused<#x_scope2>[#decl_x])
// The construct for "y" is in b.inc, which line 47 includes. Its branches
// start at 2:3 and 4:3 there.
#y_include = loc("t.f90":47:1)
#y_sel = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>["b.inc":1:3, #y_include])
#y_scope1 = #fir.lexical_scope<loc = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>["b.inc":2:3, #y_include])>
#y_scope2 = #fir.lexical_scope<loc = loc(fused<#fir<loc_kind_array[<base>, <inclusion>]>>["b.inc":4:3, #y_include])>
#y_branch1 = loc(fused<#y_scope1>[fused<#fir<loc_kind_array[<base>, <inclusion>]>>["b.inc":3:5, #y_include]])
#y_branch2 = loc(fused<#y_scope2>[fused<#fir<loc_kind_array[<base>, <inclusion>]>>["b.inc":5:5, #y_include]])
#decl_y1 = loc(fused<#y_scope1>[#decl_y])
#decl_y2 = loc(fused<#y_scope2>[#decl_y])
#inc_after = loc("t.f90":48:3)
#inc_ret = loc("t.f90":49:1)

// CHECK: func.func @_QPinc
// CHECK: ^bb1:
// CHECK: fir.load {{.*}} loc(#[[X_INCLUDED:loc[0-9]+]])
// CHECK: ^bb4:
// CHECK: fir.load {{.*}} loc(#[[Y_BRANCH1:loc[0-9]+]])
// CHECK: ^bb5:
// CHECK: fir.load {{.*}} loc(#[[Y_BRANCH2:loc[0-9]+]])
// CHECK: ^bb6:
// CHECK-NEXT: fir.load {{.*}} loc(#[[INC_AFTER:loc[0-9]+]])

// Code that a branch includes is in the scope of the branch.
// CHECK-DAG: #[[INC:.+]] = #llvm.di_subprogram<{{.*}}name = "inc"{{.*}}>
// CHECK-DAG: #[[X1:.+]] = #llvm.di_lexical_block<scope = #[[INC]], {{.*}}line = 43, column = 5>
// CHECK-DAG: #[[X_INCLUDED]] = loc(fused<#[[X1]]>[#{{loc[0-9]+}}])

// The scopes of the branches of a construct in an included file are distinct,
// positioned in that file.
// CHECK-DAG: #[[B_FILE:.+]] = #llvm.di_file<"b.inc" in "">
// CHECK-DAG: #[[Y1:.+]] = #llvm.di_lexical_block<scope = #[[INC]], file = #[[B_FILE]], line = 2, column = 3>
// CHECK-DAG: #[[Y2:.+]] = #llvm.di_lexical_block<scope = #[[INC]], file = #[[B_FILE]], line = 4, column = 3>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[Y1]], name = "y"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[Y2]], name = "y"
// CHECK-DAG: #[[Y_BRANCH1]] = loc(fused<#[[Y1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[Y_BRANCH2]] = loc(fused<#[[Y2]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[INC_AFTER]] = loc("t.f90":48:3)

//--- inl.fir
module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPinl(%arg0: !fir.ref<i32>) {
    %true = arith.constant true loc(#inl)
    %0 = fircg.ext_declare %arg0 uniq_name("_QFinlEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p_inl)
    cf.cond_br %true, ^bb1, ^bb2 loc(#inl_sel)
  ^bb1:
    %1 = fircg.ext_declare %arg0 uniq_name("_QFinlEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p_inl1)
    // Inlined twice from a procedure that associates "r".
    %2 = fircg.ext_declare %arg0 uniq_name("_QFcalleeEr") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#inlined_decl_r1)
    %3 = fir.load %2 : !fir.ref<i32> loc(#inlined1)
    %4 = fircg.ext_declare %arg0 uniq_name("_QFcalleeEr") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#inlined_decl_r2)
    %5 = fir.load %4 : !fir.ref<i32> loc(#inlined2)
    cf.br ^bb3 loc(#inl_branch1)
  ^bb2:
    %6 = fircg.ext_declare %arg0 uniq_name("_QFinlEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p_inl2)
    cf.br ^bb3 loc(#inl_branch2)
  ^bb3:
    %7 = fir.load %0 : !fir.ref<i32> loc(#inlined_after)
    return loc(#inl_ret)
  } loc(#inl)
} loc(#mod)

#mod = loc("t.f90":1:1)
#inl = loc("t.f90":50:1)
#decl_p_inl = loc("t.f90":51:14)
#inl_sel = loc("t.f90":52:3)
#inl_ret = loc("t.f90":59:1)
#p_scope1 = #fir.lexical_scope<loc = loc("t.f90":53:5)>
#p_scope2 = #fir.lexical_scope<loc = loc("t.f90":56:5)>
#inl_branch1 = loc(fused<#p_scope1>["t.f90":53:5])
#inl_branch2 = loc(fused<#p_scope2>["t.f90":56:5])
#decl_p_inl1 = loc(fused<#p_scope1>[#decl_p_inl])
#decl_p_inl2 = loc(fused<#p_scope2>[#decl_p_inl])
// The callee associates "r" in a branch at 62:5. It is inlined at 54:7 and
// 55:7 in the first branch of the construct for "p", and at 58:3 after it.
#r_scope = #fir.lexical_scope<loc = loc("t.f90":62:5)>
#callee_decl_r = loc(fused<#r_scope>["t.f90":62:16])
#callee_body = loc(fused<#r_scope>["t.f90":63:7])
#call1 = loc(fused<#p_scope1>["t.f90":54:7])
#call2 = loc(fused<#p_scope1>["t.f90":55:7])
#call3 = loc("t.f90":58:3)
#inlined_decl_r1 = loc(callsite(#callee_decl_r at #call1))
#inlined1 = loc(callsite(#callee_body at #call1))
#inlined_decl_r2 = loc(callsite(#callee_decl_r at #call2))
#inlined2 = loc(callsite(#callee_body at #call2))
#inlined_after = loc(callsite(#callee_body at #call3))

// CHECK: func.func @_QPinl
// CHECK: ^bb1:
// CHECK: fir.load {{.*}} loc(#[[INLINED1:loc[0-9]+]])
// CHECK: fir.load {{.*}} loc(#[[INLINED2:loc[0-9]+]])
// CHECK: ^bb3:
// CHECK-NEXT: fir.load {{.*}} loc(#[[INLINED_AFTER:loc[0-9]+]])

// Inlined code is in the scope of its call, not in the scope that the callee
// tags it with.
// CHECK-DAG: #[[INL:.+]] = #llvm.di_subprogram<{{.*}}name = "inl"{{.*}}>
// CHECK-DAG: #[[P_INL1:.+]] = #llvm.di_lexical_block<scope = #[[INL]], {{.*}}line = 53, column = 5>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[P_INL1]], name = "r"
// CHECK-DAG: #[[INLINED1]] = loc(fused<#[[P_INL1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[INLINED2]] = loc(fused<#[[P_INL1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[INLINED_AFTER]] = loc(callsite(
