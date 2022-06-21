// RUN: %not_todo_cmd fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s 2>&1 | FileCheck %s

// Test `fir.select_case` conversion to llvm with character type.
// Not implemented yet.

func.func @select_case_charachter(%arg0: !fir.char<2, 10>, %arg1: !fir.char<2, 10>, %arg2: !fir.char<2, 10>) {
// CHECK: not yet implemented: fir.select_case codegen with character type
  fir.select_case %arg0 : !fir.char<2, 10> [#fir.point, %arg1, ^bb1,
                                            #fir.point, %arg2, ^bb2,
                                            unit, ^bb3]
^bb1:
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb3
^bb2:
  %c2_i32 = arith.constant 2 : i32
  cf.br ^bb3
^bb3:
  return
}
