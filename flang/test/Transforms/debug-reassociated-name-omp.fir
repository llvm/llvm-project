// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

// Test the scopes of names declared more than once in OpenMP code.

module attributes {dlti.dl_spec = #dlti.dl_spec<>} {
  omp.private {type = private} @_QForphanEi_private_i32 : i32 loc(#orphan_decl)
  omp.private {type = linear} @_QFlinEk_linear_i32 : i32 loc(#lin_decl)
  omp.private {type = private} @_QFbrloopEi_private_i32 : i32 loc(#brl_decl)

  func.func @_QPorphan() {
    %c1 = arith.constant 1 : i32 loc(#orphan)
    %c8 = arith.constant 8 : i32 loc(#orphan)
    %0 = fir.alloca i32 loc(#orphan_decl)
    %1 = fircg.ext_declare %0 uniq_name("_QForphanEi") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#orphan_decl)
    omp.wsloop private(@_QForphanEi_private_i32 %1 -> %arg0 : !fir.ref<i32>) {
      omp.loop_nest (%iv) : i32 = (%c1) to (%c8) inclusive step (%c1) {
        %2 = fircg.ext_declare %arg0 uniq_name("_QForphanEi") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#orphan_decl)
        fir.store %iv to %2 : !fir.ref<i32> loc(#orphan_body)
        omp.yield loc(#orphan_body)
      } loc(#orphan_loop)
    } loc(#orphan_loop)
    return loc(#orphan_ret)
  } loc(#orphan)

  func.func @_QPlin() {
    %c1 = arith.constant 1 : i32 loc(#lin)
    %c8 = arith.constant 8 : i32 loc(#lin)
    %0 = fir.alloca i32 loc(#lin_decl)
    %1 = fircg.ext_declare %0 uniq_name("_QFlinEk") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#lin_decl)
    omp.simd linear(@_QFlinEk_linear_i32 %1 -> %arg0 : !fir.ref<i32> = %c1 : i32) linear_var_types([i32]) {
      omp.loop_nest (%iv) : i32 = (%c1) to (%c8) inclusive step (%c1) {
        %2 = fircg.ext_declare %arg0 uniq_name("_QFlinEk") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#lin_decl)
        %3 = fir.load %2 : !fir.ref<i32> loc(#lin_body)
        omp.yield loc(#lin_body)
      } loc(#lin_loop)
    } loc(#lin_loop)
    return loc(#lin_ret)
  } loc(#lin)

  func.func @_QPtr(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>) {
    %true = arith.constant true loc(#tr)
    %0 = fircg.ext_declare %arg0 uniq_name("_QFtrEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_decl_x)
    %1 = fircg.ext_declare %arg1 uniq_name("_QFtrEy") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_decl_y)
    cf.cond_br %true, ^bb1, ^bb2 loc(#tr_sel)
  ^bb1:
    %2 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_branch1)
    %3 = fircg.ext_declare %2 uniq_name("_QFtrEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_decl_x1)
    %4 = omp.map.info var_ptr(%1 : !fir.ref<i32>, i32) map_clauses(from) capture(ByRef) name("y") -> !fir.ref<i32> loc(#tr_decl_y1)
    omp.target kernel_type(generic) map_entries(%4 -> %arg2 : !fir.ref<i32>) {
      %5 = fircg.ext_declare %arg2 uniq_name("_QFtrEy") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_decl_y1)
      %c1_0 = arith.constant 1 : i32 loc(#tr_body)
      fir.store %c1_0 to %5 : !fir.ref<i32> loc(#tr_body)
      omp.terminator loc(#tr_target)
    } loc(#tr_target)
    cf.br ^bb3 loc(#tr_branch1)
  ^bb2:
    %6 = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_branch2)
    %7 = fircg.ext_declare %6 uniq_name("_QFtrEx") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#tr_decl_x2)
    cf.br ^bb3 loc(#tr_branch2)
  ^bb3:
    return loc(#tr_ret)
  } loc(#tr)

  func.func @_QPbrloop() {
    %true = arith.constant true loc(#brl)
    %c1 = arith.constant 1 : i32 loc(#brl)
    %c8 = arith.constant 8 : i32 loc(#brl)
    %0 = fir.alloca i32 loc(#brl_decl)
    %1 = fircg.ext_declare %0 uniq_name("_QFbrloopEi") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#brl_decl)
    cf.cond_br %true, ^bb1, ^bb2 loc(#brl_sel)
  ^bb1:
    omp.wsloop private(@_QFbrloopEi_private_i32 %1 -> %arg0 : !fir.ref<i32>) {
      omp.loop_nest (%iv) : i32 = (%c1) to (%c8) inclusive step (%c1) {
        %2 = fircg.ext_declare %arg0 uniq_name("_QFbrloopEi") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#brl_decl1)
        fir.store %iv to %2 : !fir.ref<i32> loc(#brl_body)
        omp.yield loc(#brl_body)
      } loc(#brl_loop)
    } loc(#brl_loop)
    %3 = fir.load %1 : !fir.ref<i32> loc(#brl_after_loop)
    cf.br ^bb2 loc(#brl_branch)
  ^bb2:
    return loc(#brl_ret)
  } loc(#brl)
} loc(#mod)

#mod = loc("t.f90":1:1)

#orphan = loc("t.f90":2:1)
#orphan_decl = loc("t.f90":3:14)
#orphan_loop = loc("t.f90":5:3)
#orphan_body = loc("t.f90":6:5)
#orphan_ret = loc("t.f90":9:1)

#lin = loc("t.f90":11:1)
#lin_decl = loc("t.f90":12:17)
#lin_loop = loc("t.f90":14:3)
#lin_body = loc("t.f90":15:5)
#lin_ret = loc("t.f90":17:1)

#tr = loc("t.f90":40:1)
#tr_decl_x = loc("t.f90":41:26)
#tr_decl_y = loc("t.f90":42:27)
#tr_sel = loc("t.f90":43:3)
#tr_ret = loc("t.f90":51:1)
#tr_scope1 = #fir.lexical_scope<loc = loc("t.f90":44:3)>
#tr_scope2 = #fir.lexical_scope<loc = loc("t.f90":48:3)>
#tr_branch1 = loc(fused<#tr_scope1>["t.f90":44:3])
#tr_decl_x1 = loc(fused<#tr_scope1>[#tr_decl_x])
#tr_decl_y1 = loc(fused<#tr_scope1>[#tr_decl_y])
#tr_target = loc(fused<#tr_scope1>["t.f90":45:11])
#tr_body = loc(fused<#tr_scope1>["t.f90":46:7])
#tr_branch2 = loc(fused<#tr_scope2>["t.f90":48:3])
#tr_decl_x2 = loc(fused<#tr_scope2>[#tr_decl_x])

#brl = loc("t.f90":60:1)
#brl_decl = loc("t.f90":61:14)
#brl_sel = loc("t.f90":62:3)
#brl_ret = loc("t.f90":70:1)
// The loop is in a branch, which declares no name of its own.
#brl_scope = #fir.lexical_scope<loc = loc("t.f90":63:5)>
#brl_branch = loc(fused<#brl_scope>["t.f90":63:5])
#brl_loop = loc(fused<#brl_scope>["t.f90":65:7])
#brl_decl1 = loc(fused<#brl_scope>[#brl_decl])
#brl_body = loc(fused<#brl_scope>["t.f90":66:9])
#brl_after_loop = loc(fused<#brl_scope>["t.f90":68:7])

// CHECK: func.func @_QPorphan
// CHECK: fir.store {{.*}} loc(#[[ORPHAN_BODY:loc[0-9]+]])
// CHECK: func.func @_QPlin
// CHECK: fir.load {{.*}} loc(#[[LIN_BODY:loc[0-9]+]])
// CHECK: func.func @_QPtr
// CHECK: fir.store {{.*}} loc(#[[TR_BODY:loc[0-9]+]])
// CHECK-NEXT: omp.terminator
// CHECK-NEXT: } loc(#[[TR_TARGET:loc[0-9]+]])
// CHECK: func.func @_QPbrloop
// CHECK: fir.store {{.*}} loc(#[[BRL_BODY:loc[0-9]+]])
// CHECK: fir.load {{.*}} loc(#[[BRL_AFTER_LOOP:loc[0-9]+]])

// The private loop variable of a worksharing loop is scoped in the loop body.
// CHECK-DAG: #[[ORPHAN:.+]] = #llvm.di_subprogram<{{.*}}name = "orphan"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[ORPHAN]], name = "i"
// CHECK-DAG: #[[DO:.+]] = #llvm.di_lexical_block<scope = #[[ORPHAN]], {{.*}}line = 6, column = 5>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[DO]], name = "i"
// CHECK-DAG: #[[ORPHAN_BODY]] = loc(fused<#[[DO]]>[#{{loc[0-9]+}}])

// Likewise for a linear variable of a SIMD loop.
// CHECK-DAG: #[[LIN:.+]] = #llvm.di_subprogram<{{.*}}name = "lin"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[LIN]], name = "k"
// CHECK-DAG: #[[SIMD:.+]] = #llvm.di_lexical_block<scope = #[[LIN]], {{.*}}line = 15, column = 5>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[SIMD]], name = "k"
// CHECK-DAG: #[[LIN_BODY]] = loc(fused<#[[SIMD]]>[#{{loc[0-9]+}}])

// A target region in a branch keeps its own subprogram.
// CHECK-DAG: #[[TR:.+]] = #llvm.di_subprogram<{{.*}}name = "tr"
// CHECK-DAG: #[[RANK1:.+]] = #llvm.di_lexical_block<scope = #[[TR]], {{.*}}line = 44, column = 3>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[RANK1]], name = "x"
// CHECK-DAG: #[[TARGET:.+]] = #llvm.di_subprogram<{{.*}}name = "__omp_offloading_{{.*}}_QPtr_l45"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[TARGET]], name = "y"
// CHECK-DAG: #[[TR_TARGET]] = loc(fused<#[[TARGET]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[TR_BODY]] = loc(fused<#fir.lexical_scope<{{.*}}>>[#{{loc[0-9]+}}])

// The private copy of a loop in a branch is scoped in the loop body, not in
// the branch.
// CHECK-DAG: #[[BRL:.+]] = #llvm.di_subprogram<{{.*}}name = "brloop"
// CHECK-DAG: #llvm.di_local_variable<scope = #[[BRL]], name = "i"
// CHECK-DAG: #[[BRL_DO:.+]] = #llvm.di_lexical_block<scope = #[[BRL]], {{.*}}line = 66, column = 9>
// CHECK-DAG: #llvm.di_local_variable<scope = #[[BRL_DO]], name = "i"
// CHECK-DAG: #[[BRL_BODY]] = loc(fused<#[[BRL_DO]]>[#{{loc[0-9]+}}])
// CHECK-DAG: #[[BRL_AFTER_LOOP]] = loc(fused<#fir.lexical_scope<{{.*}}>>[#{{loc[0-9]+}}])
