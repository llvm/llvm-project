// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fcxx-exceptions -fexceptions -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++14 -fcxx-exceptions -fexceptions -emit-llvm %s -o %t-ogcg.ll
// RUN: FileCheck --input-file=%t-ogcg.ll --check-prefix=OGCG %s

// Regression test: try/catch inside a do-while loop with a conditional
// return in the catch handler was triggering an assertion in
// flattenCatchHandler.  The assertion required end_catch to be
// reachable through only cir.br predecessors, but flattened cir.if
// produces conditional branches on the path.

namespace std { class bad_alloc {}; }
void may_throw();
bool check();

int test_catch_in_loop() {
  int tries = 0;
  do {
    try {
      may_throw();
    } catch (std::bad_alloc &) {
      tries += 1;
      if (tries > 10)
        return -1;
    }
  } while (!check());
  return 0;
}

// Verify the catch handler has begin_catch, end_catch, and the
// cleanup scope with the conditional return.
// CIR: cir.func {{.*}} @_Z18test_catch_in_loopv
// CIR: cir.try
// CIR: cir.call @_Z9may_throwv
// CIR: cir.begin_catch
// CIR: cir.cleanup.scope
// CIR: cir.if
// CIR: cir.return
// CIR: cir.end_catch

// LLVM: define {{.*}} @_Z18test_catch_in_loopv()
// LLVM: invoke void @_Z9may_throwv()
// LLVM: landingpad
// LLVM: call ptr @__cxa_begin_catch
// LLVM: call void @__cxa_end_catch()
// LLVM: ret i32

// OGCG: define {{.*}} @_Z18test_catch_in_loopv()
// OGCG: invoke void @_Z9may_throwv()
// OGCG: landingpad
// OGCG: call ptr @__cxa_begin_catch
// OGCG: call void @__cxa_end_catch()
// OGCG: ret i32
