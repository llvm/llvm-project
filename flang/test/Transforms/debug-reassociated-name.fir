// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test the scopes of a name declared in more than one block.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  func.func @_QPreassoc(%arg0: !fir.ref<i32>) {
    %true = arith.constant true loc(#func)
    %ds = fir.undefined !fir.dscope loc(#func)
    %0 = fircg.ext_declare %arg0 dummy_scope %ds arg 1 uniq_name("_QFreassocEsel") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#decl_sel)
    cf.cond_br %true, ^bb1, ^bb2 loc(#sel)
  ^bb1:
    %1 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#branch1)
    %2 = fircg.ext_declare %1 uniq_name("_QFreassocEsel") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_sel)
    cf.br ^bb3 loc(#branch1)
  ^bb2:
    %3 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#branch2)
    %4 = fircg.ext_declare %3 uniq_name("_QFreassocEsel") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_sel)
    cf.br ^bb3 loc(#branch2)
  ^bb3:
    // Declared once.
    %5 = fircg.ext_declare %arg0 uniq_name("_QFreassocEonce") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_once)
    return loc(#ret)
  } loc(#func)

  func.func @_QPnest(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
    %true = arith.constant true loc(#nest)
    %0 = fircg.ext_declare %arg0 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p)
    %1 = fircg.ext_declare %arg1 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q)
    cf.cond_br %true, ^bb1, ^bb5 loc(#sel_p)
  ^bb1:
    %2 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#p_branch1)
    %3 = fircg.ext_declare %2 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p)
    cf.cond_br %true, ^bb2, ^bb3 loc(#sel_q)
  ^bb2:
    %4 = fir.convert %arg1 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#q_branch1)
    %5 = fircg.ext_declare %4 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q)
    cf.br ^bb4 loc(#q_branch1)
  ^bb3:
    %6 = fir.convert %arg1 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#q_branch2)
    %7 = fircg.ext_declare %6 uniq_name("_QFnestEq") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_q)
    cf.br ^bb4 loc(#q_branch2)
  ^bb4:
    %8 = fir.load %3 : !fir.ref<i32> loc(#loop)
    cf.cond_br %true, ^bb4, ^bb6 loc(#loop)
  ^bb5:
    %9 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#p_branch2)
    %10 = fircg.ext_declare %9 uniq_name("_QFnestEp") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#decl_p)
    cf.br ^bb6 loc(#p_branch2)
  ^bb6:
    return loc(#nest_ret)
  } loc(#nest)
} loc(#mod)

#mod = loc("t.f90":1:1)
#func = loc("t.f90":3:1)
#decl_once = loc("t.f90":4:14)
// All the declarations of "sel" share a location.
#decl_sel = loc("t.f90":5:14)
#sel = loc("t.f90":7:3)
#branch1 = loc("t.f90":8:5)
#branch2 = loc("t.f90":8:20)
#ret = loc("t.f90":12:1)

#nest = loc("t.f90":20:1)
#decl_p = loc("t.f90":21:14)
#decl_q = loc("t.f90":21:17)
#sel_p = loc("t.f90":22:3)
#p_branch1 = loc("t.f90":23:5)
#sel_q = loc("t.f90":24:5)
#q_branch1 = loc("t.f90":25:7)
#q_branch2 = loc("t.f90":27:7)
#loop = loc("t.f90":30:7)
#p_branch2 = loc("t.f90":33:5)
#nest_ret = loc("t.f90":35:1)

// CHECK: func.func @_QPnest
// CHECK: ^bb4:
// CHECK-NEXT: fir.load {{.*}} loc(#[[LOOP:loc[0-9]+]])
// CHECK: ^bb6:
// CHECK: return loc(#[[NEST_RET:loc[0-9]+]])

// CHECK-DAG: #[[SP:.+]] = #llvm.di_subprogram<{{.*}}name = "reassoc"{{.*}}>

// A declaration in the entry block stays in the procedure.
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SP]], name = "sel"{{.*}}arg = 1

// Each branch has its own scope, positioned at the branch, even on one line.
// CHECK-DAG: #[[B1:.+]] = #llvm.di_lexical_block<scope = #[[SP]], {{.*}}line = 8, column = 5>
// CHECK-DAG: #[[B2:.+]] = #llvm.di_lexical_block<scope = #[[SP]], {{.*}}line = 8, column = 20>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[B1]], name = "sel"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[B2]], name = "sel"

// A name declared once stays in the procedure.
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SP]], name = "once"

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

// The scope of a branch covers the blocks it dominates and ends with the
// construct.
// CHECK-DAG: #[[LOOP]] = loc(fused<#[[P1]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[NEST_RET]] = loc("t.f90":35:1)
