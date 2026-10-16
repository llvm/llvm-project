// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" --mlir-print-debuginfo %s | FileCheck %s
// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=false" %s | FileCheck %s --check-prefix=NOSLOT
// RUN: sed 's/amdgcn-amd-amdhsa/x86_64-unknown-linux-gnu/' %s | fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" | FileCheck %s --check-prefix=NOSLOT

// For AMDGPU, a dummy argument that is located by the address it is passed in
// is described through a copy of that address, which is kept in a stack slot,
// when the procedure returns from its entry block.

#loc1 = loc("test.f90":1:1)
#loc3 = loc("test.f90":3:14)
#loc4 = loc("test.f90":4:14)
#loc5 = loc("test.f90":5:1)
#loc = loc("test.f90":0:0)

module attributes {dlti.dl_spec = #dlti.dl_spec<>, llvm.target_triple = "amdgcn-amd-amdhsa"} {
  func.func private @foo()

  func.func @test_dummies(%arg0: !fir.ref<i32> {fir.bindc_name = "x"} loc(#loc1), %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"} loc(#loc1)) attributes {fir.internal_name = "_QPtest_dummies"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFtest_dummiesEx") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 uniq_name("_QFtest_dummiesEy") : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?xi32>> loc(#loc3)
    %3 = fir.alloca i32 loc(#loc4)
    %4 = fircg.ext_declare %3 uniq_name("_QFtest_dummiesElocal") : (!fir.ref<i32>) -> !fir.ref<i32> loc(#loc4)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  func.func @test_branch(%arg0: !fir.ref<i32> {fir.bindc_name = "x"} loc(#loc1), %arg1: i1 loc(#loc1)) attributes {fir.internal_name = "_QPtest_branch"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFtest_branchEx") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    cf.cond_br %arg1, ^bb1, ^bb2
  ^bb1:
    fir.call @foo() : () -> ()
    cf.br ^bb2
  ^bb2:
    return loc(#loc5)
  } loc(#loc1)
} loc(#loc)

// CHECK-LABEL: func.func @test_dummies(
// CHECK-NEXT:    %[[SLOT:.*]] = fir.alloca !fir.llvm_ptr<i32> loc(#[[UNK:loc[0-9]+]])
// CHECK-NEXT:    fir.undefined !fir.dscope
// CHECK-NEXT:    %[[ADDR:.*]] = fir.convert %arg0 : (!fir.ref<i32>) -> !fir.llvm_ptr<i32> loc(#[[UNK]])
// CHECK-NEXT:    fir.store %[[ADDR]] to %[[SLOT]] : !fir.ref<!fir.llvm_ptr<i32>> loc(#[[UNK]])
// CHECK-NEXT:    %[[LOAD:.*]] = fir.load %[[SLOT]] : !fir.ref<!fir.llvm_ptr<i32>> loc(#[[UNK]])
// CHECK-NEXT:    %[[COPY:.*]] = fir.convert %[[LOAD]] : (!fir.llvm_ptr<i32>) -> !fir.ref<i32> loc(#[[UNK]])
// CHECK-NEXT:    fircg.ext_declare %[[COPY]] dummy_scope %{{.*}} arg 1 uniq_name("_QFtest_dummiesEx") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#[[X_LOC:loc[0-9]+]])
// A dummy argument passed by descriptor is described as before.
// CHECK-NEXT:    fircg.ext_declare %arg1 dummy_scope
// CHECK:         fircg.ext_declare %{{.*}} uniq_name("_QFtest_dummiesElocal")
// CHECK:         fir.call @foo()
// CHECK-NEXT:    fir.fake_use %[[COPY]]
// CHECK-NEXT:    fir.fake_use %arg1
// CHECK-NEXT:    return

// The fake use before the return already keeps the argument alive outside the
// entry block.
// CHECK-LABEL: func.func @test_branch(
// CHECK-NOT:     fir.llvm_ptr
// CHECK:         fircg.ext_declare %arg0 dummy_scope %{{.*}} arg 1 uniq_name("_QFtest_branchEx") : {{.*}} loc(#[[BR_LOC:loc[0-9]+]])
// CHECK-NOT:     fir.llvm_ptr
// CHECK:       ^bb2:
// CHECK-NEXT:    fir.fake_use %arg0
// CHECK-NEXT:    return

// CHECK:       #[[UNK]] = loc(unknown)
// CHECK:       #[[X_SRC:loc[0-9]+]] = loc("test.f90":3:14)
// CHECK:       #[[BR_SP:di_subprogram[0-9]*]] = #llvm.di_subprogram<{{.*}}name = "test_branch"
// CHECK:       #[[BR_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<scope = #[[BR_SP]], name = "x"{{.*}}arg = 1
// CHECK:       #[[X_SP:di_subprogram[0-9]*]] = #llvm.di_subprogram<{{.*}}name = "test_dummies"
// CHECK:       #[[X_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<scope = #[[X_SP]], name = "x"{{.*}}arg = 1
// CHECK:       #[[BR_LOC]] = loc(fused<#[[BR_VAR]]>[#[[X_SRC]]])
// CHECK:       #[[X_LOC]] = loc(fused<#[[X_VAR]]>[#[[X_SRC]]])

// NOSLOT-LABEL: func.func @test_dummies(
// NOSLOT-NOT:     fir.llvm_ptr
// NOSLOT:         fircg.ext_declare %arg0 dummy_scope %{{.*}} arg 1
// NOSLOT-LABEL: func.func @test_branch(
// NOSLOT-NOT:     fir.llvm_ptr
