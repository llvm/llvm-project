// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// CHECK: _Z23unreachable_after_breaki
void unreachable_after_break(int a) {
  switch(a) {
  case 0:
    break;
    break;
    int x = 1;
  }
  // cir.switch
  //   cir.case(equal, [#cir.int<0> : !s32i]) {
  //     cir.break
  //   ^bb{{.*}}:  // no predecessors
  //     cir.break
  //   ^bb{{.*}}:  // no predecessors
  //     %[[CONST:.*]] = cir.const #cir.int<1> : !s32i
  //     cir.store align(4) {{.*}}, %[[CONST]]
  //     cir.yield
}

// CHECK: _Z24unreachable_after_returni
int unreachable_after_return(int a) {
  switch (a) {
  case 0:
    return 0;
    return 1;
    int x = 3;
  }
  return 2;
  // cir.switch
  //   cir.case(equal, [#cir.int<0> : !s32i]) {
  //     %[[CONST_ZERO:.*]] = cir.const #cir.int<0> : !s32i
  //     cir.store {{.*}}, %[[CONST_ZERO]]
  //     cir.br ^bb1
  //   ^bb1:  // 2 preds: ^bb0, ^bb2
  //     cir.load
  //     cir.return
  //   ^bb2:  // no predecessors
  //     %[[CONST_ONE:.*]] = cir.const #cir.int<1> : !s32i
  //     cir.store %[[CONST_ONE]]
  //     cir.br ^bb1
  //   ^bb3:  // no predecessors
  //     %[[CONST_THREE:.*]] = cir.const #cir.int<3> : !s32i
  //     cir.store align(4) %[[CONST_THREE]]
  //     cir.yield
  //   }
}
