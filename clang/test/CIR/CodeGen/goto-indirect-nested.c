// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// A `goto *p` inside a nested scope, jumping to a top-level label.
int nested_goto(int x) {
  void *p = &&done;
  if (x)
    goto *p;
done:
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @nested_goto
// CIR:   %[[P:.*]] = cir.alloca "p"
// CIR:   cir.block_address <@nested_goto, "done"> : !cir.ptr<!void>
// CIR:   cir.scope {
// CIR:     cir.if %{{.*}} {
// CIR:       %[[T:.*]] = cir.load align(8) %[[P]]
// CIR:       cir.indirect_goto %[[T]] : !cir.ptr<!void>
// CIR:     }
// CIR:   }
// CIR:   cir.label "done"

// LLVM-LABEL: define dso_local i32 @nested_goto
// LLVM:   store ptr blockaddress(@nested_goto, %[[DONE:[0-9]+]]), ptr %{{.*}}, align 8
// LLVM:   indirectbr ptr %{{.*}}, [label %[[DONE]]]

// OGCG-LABEL: define dso_local i32 @nested_goto
// OGCG:   store ptr blockaddress(@nested_goto, %[[DONE:.*]]), ptr %{{.*}}, align 8
// OGCG:   indirectbr ptr %{{.*}}, [label %[[DONE]]]

// A top-level `goto *p` whose target label sits inside a nested scope.
int nested_label(int x) {
  void *p;
  if (x) {
  inner:
    return 1;
  }
  p = &&inner;
  goto *p;
}

// CIR-LABEL: cir.func {{.*}} @nested_label
// CIR:   cir.scope {
// CIR:     cir.if %{{.*}} {
// CIR:       cir.label "inner"
// CIR:     }
// CIR:   }
// CIR:   cir.block_address <@nested_label, "inner"> : !cir.ptr<!void>
// CIR:   %[[T:.*]] = cir.load align(8)
// CIR:   cir.indirect_goto %[[T]] : !cir.ptr<!void>

// LLVM-LABEL: define dso_local i32 @nested_label
// LLVM:   store ptr blockaddress(@nested_label, %[[INNER:[0-9]+]]), ptr %{{.*}}, align 8
// LLVM:   indirectbr ptr %{{.*}}, [label %[[INNER]]]

// OGCG-LABEL: define dso_local i32 @nested_label
// OGCG:   indirectbr ptr %{{.*}}, [label %[[INNER:.*]]]
// OGCG:   store ptr blockaddress(@nested_label, %[[INNER]]), ptr %{{.*}}, align 8

// A `goto *p` inside a loop body.
int goto_in_loop(int n) {
  void *p = &&out;
  for (int i = 0; i < n; ++i)
    goto *p;
out:
  return n;
}

// CIR-LABEL: cir.func {{.*}} @goto_in_loop
// CIR:   cir.block_address <@goto_in_loop, "out"> : !cir.ptr<!void>
// CIR:   cir.for : cond {
// CIR:   } body {
// CIR:     %[[T:.*]] = cir.load align(8)
// CIR:     cir.indirect_goto %[[T]] : !cir.ptr<!void>
// CIR:   } step {
// CIR:   }
// CIR:   cir.label "out"

// LLVM-LABEL: define dso_local i32 @goto_in_loop
// LLVM:   store ptr blockaddress(@goto_in_loop, %[[OUT:[0-9]+]]), ptr %{{.*}}, align 8
// LLVM:   indirectbr ptr %{{.*}}, [label %[[OUT]]]

// OGCG-LABEL: define dso_local i32 @goto_in_loop
// OGCG:   store ptr blockaddress(@goto_in_loop, %[[OUT:.*]]), ptr %{{.*}}, align 8
// OGCG:   indirectbr ptr %{{.*}}, [label %[[OUT]]]

// An address-taken label as the first statement of the function.
int leading_label(int x) {
first:;
  void *p = &&first;
  if (x)
    goto *p;
  return 0;
}

// CIR-LABEL: cir.func {{.*}} @leading_label
// CIR:   cir.br ^bb1
// CIR: ^bb1:
// CIR:   cir.label "first"
// CIR:   cir.block_address <@leading_label, "first"> : !cir.ptr<!void>
// CIR:   cir.indirect_goto %{{.*}} : !cir.ptr<!void>

// LLVM-LABEL: define dso_local i32 @leading_label
// LLVM:   br label %[[FIRST:[0-9]+]]
// LLVM: [[FIRST]]:
// LLVM:   store ptr blockaddress(@leading_label, %[[FIRST]]), ptr %{{.*}}, align 8
// LLVM:   indirectbr ptr %{{.*}}, [label %[[FIRST]]]

// OGCG-LABEL: define dso_local i32 @leading_label
// OGCG:   br label %[[FIRST:.*]]
// OGCG: [[FIRST]]:
// OGCG:   store ptr blockaddress(@leading_label, %[[FIRST]]), ptr %{{.*}}, align 8
// OGCG:   indirectbr ptr %{{.*}}, [label %[[FIRST]]]
