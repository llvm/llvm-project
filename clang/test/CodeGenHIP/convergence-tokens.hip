// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn -fconvergence-control -Wno-convergence -emit-llvm -disable-llvm-optzns -fcuda-is-device -o - %s | FileCheck %s


// NOTE:
// =====
// The following tests are of particular interest:
// - jump_into_unreachable_nest
// - backwards_jump_into_nest
// - forever_loops
// - backwards_inside_label
// - switch_backwards
// - backwards_conditional
// - duffs_device

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

// CHECK-LABEL: @_Z14jump_into_nesti
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: if.end:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK2]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then2
// CHECK:   [[TOK4:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK4]]) ]
// CHECK: do.end:                                           ; preds = %do.cond
// CHECK:   [[TOK7:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK7]]) ]

__device__ int jump_into_nest(int cond) {
  int i = 0;

  if (cond > 1) {
    goto jumptarget;
  }

  foo();

  while (true) {
    foo();
    if (cond > 1) {
      do {
        i++;
        foo();
      jumptarget:
        i++;
      } while (true);
      foo();
    }
  }

  return i;
}

// CHECK-LABEL: @_Z26jump_into_unreachable_nesti
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK2]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then
// CHECK:   [[TOK4:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK4]]) ]
// CHECK: do.end:                                           ; preds = %do.cond
// CHECK:   [[TOK7:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK7]]) ]

// A function with a nest of reducible forever loops, where the entry is from
// the label ``jumptarget`` since they are unreachable with sequential control
// flow. The detected cycles are inside out ... the do-while loop is the parent
// of the while loop. What's even more weird is that the inner loop is actually
// unreachable because the condition for the do-while never branches to it.
__device__ int jump_into_unreachable_nest(int cond) {
  int i = 0;

  foo();

  goto jumptarget;

  while (true) {
    foo();
    if (cond > 1) {
      do {
        i++;
        foo();
      jumptarget:
        i++;
      } while (true);
      foo();
    }
  }

  return i;
}

// CHECK-LABEL: @_Z24backwards_jump_into_nesti
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:                                       ; preds = %if.end, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then
// CHECK:   [[TOK6:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK6]]) ]
// CHECK: do.end:                                           ; preds = %do.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]

// The outer while-loop dominates the backwards jump, and hence remains
// reducible. But the do-while loop does not dominate and hence becomes an
// irreducible cycle with two entries: the blocks %do.body and %jumptarget.
__device__ int backwards_jump_into_nest(int cond1, int cond2, int cond3, int cond4) {
  int i = 0;

  while (i < cond1) {
    foo();
    if (cond2 > i) {
      do {
        i++;
        foo();
      jumptarget:
        i++;
      } while (cond3 > i);
      foo();
    }
  }

  if (cond4 > i) {
    goto jumptarget;
  }

  return i;
}

// CHECK-LABEL: @_Z17forward_all_kindsiiiii
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.body:                                         ; preds = %for.cond
// CHECK:   [[TOK4:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK4]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   [[TOK8:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK8]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then6
// CHECK:   [[TOK10:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK10]]) ]
// CHECK: while.cond10:                                     ; preds = %while.body12, %if.else
// CHECK:   [[TOK15:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK: while.body12:                                     ; preds = %while.cond10
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK15]]) ]

// All loops become irreducible due to the jump into the nest.
__device__ int forward_all_kinds(int cond1, int cond2, int cond3, int cond4, int cond5) {
  int i = 0;

  if (cond1 < 0)
    goto jumptarget;

  for (; i < cond5;) {
    foo();
    if (cond2 != 0) {
      while (i < cond3) {
        foo();
        if (cond4 > 1) {
          do {
            foo();
            i++;
          jumptarget:
            i++;
          } while (i < cond4);
        }
      }
    } else {
      while (i < cond3) {
        foo();
        i++;
      }
    }
  }
  return i;
}

// CHECK-LABEL: @_Z13forever_loopsi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:                                         ; preds = %if.end
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK2]]) ]
// CHECK: jumptarget:                                       ; preds = %while.body, %if.then
// CHECK:   [[TOK3:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK3]]) ]

// The outer for-loop is "interesting". The frontend eliminates its backedge
// effectively making it a non-loop, since control never exits the inner
// while-loop.
__device__ int forever_loops(int cond1) {
  int i = 0;

  if (cond1 < 0)
    goto jumptarget;

  for (;;) {
    foo();
    while (true) {
    jumptarget:
      i++;
      foo();
    }
  }

  return i;
}

// CHECK-LABEL:  @_Z12nest_to_nesti
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:                                         ; preds = %if.end3, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: while.cond:                                       ; preds = %if.end, %if.then
// CHECK:   [[TOK3:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK3]]) ]
// CHECK: while.body5:                                      ; preds = %while.cond4
// CHECK:   [[TOK6:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK6]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then7
// CHECK:   [[TOK8:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK8]]) ]

__device__ int nest_to_nest(int cond) {
  int i = 0;

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget;
        }
      }
    }
  }

  while (true) {
    foo();
    if (cond > 1) {
      do {
        foo();
        i++;
      jumptarget:
        i++;
      } while (true);
    }
  }

  return i;
}

// CHECK-LABEL: @_Z22backwards_nest_to_nesti
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:                                       ; preds = %if.end, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %if.then
// CHECK:   [[TOK3:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK1]]) ]
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK3]]) ]

__device__ int backwards_nest_to_nest(int cond) {
  int i = 0;

  while (true) {
    foo();
    if (cond > 1) {
      do {
        foo();
        i++;
      jumptarget:
        i++;
      } while (true);
    }
  }

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget;
        }
      }
    }
  }

  return i;
}

// CHECK-LABEL: @_Z17backwards_and_outi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: if.then:                                          ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: jumptarget:                                       ; preds = %if.then4, %if.then
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK2]]) ]
// CHECK: for.cond:                                         ; preds = %if.end6, %if.end
// CHECK:   [[TOK4:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK4]]) ]
// CHECK: while.cond:                                       ; preds = %if.end5, %if.then2
// CHECK:   [[TOK6:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK4]]) ]
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK6]]) ]

__device__ int backwards_and_out(int cond) {
  int i = 0;

  if (cond > 1) {
    foo();
  jumptarget:
    foo();
    i++;
  }

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget;
        }
      }
    }
  }

  return i;
}

// CHECK-LABEL: @_Z21backwards_inside_loopi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:                                         ; preds = %if.end, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: jumptarget:                                       ; preds = %if.then, %for.cond
// CHECK:   [[TOK3:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK3]]) ]

__device__ int backwards_inside_loop(int cond) {
  int i = 0;
  int j = 0;
  int k = 0;
  for (;;) {
    foo();
    i++;
  jumptarget:
    foo();
    j++;
    k++;
    if (cond > 5)
      goto jumptarget;
  }
  return i + j + k;
}

// CHECK-LABEL: @_Z19loop_backwards_loopi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: for.cond:                                         ; preds = %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token [[TOK0]]) ]
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]
// CHECK: while.cond:                                       ; preds = %if.end, %jumptarget
// CHECK:   [[TOK4:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK4]]) ]

__device__ int loop_backwards_loop(int cond) {
  int i = 0;
  int j = 0;
  int k = 0;
  for (;;) {
    foo();
    i++;
  jumptarget:
    j++;
    while (true) {
      foo();
      k++;
      if (cond > 5)
        goto jumptarget;
    }
  }
  return i + j + k;
}

// CHECK-LABEL: @_Z22backwards_inside_labeli
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: jumptarget:                                       ; preds = %if.then, %entry
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]

// The border-case where a goto is a child of its own target label statement.
// This is actually a reducible cycle and can use the loop intrinsic if we
// strengthen the semantics of the backwards goto.
__device__ int backwards_inside_label(int cond) {
  int i = 0;
  int j = 0;
  jumptarget: {
    foo();
    i++;
    j++;
    if (cond > 5)
      goto jumptarget;
  }
  return i + j;
}

// CHECK-LABEL: @_Z29backwards_label_inside_branchi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: jumptarget:                                       ; preds = %if.then3, %if.then
// CHECK:   [[TOK2:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK2]]) ]

__device__ int backwards_label_inside_branch(int cond) {
  int i = 0;
  int j = 0;
  if (cond > 0) {
  jumptarget: {
      foo();
      i++;
      j++;
      if (cond > 5)
        goto jumptarget;
    }
  } else {
    j++;
  }
  return i + j;
}

// CHECK-LABEL: @_Z30backwards_inside_labelled_loopi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: while.cond:                                       ; preds = %if.end, %jumptarget
// CHECK:   [[TOK1:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK: while.body:                                       ; preds = %while.cond
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK1]]) ]

__device__ int backwards_inside_labelled_loop(int cond) {
  int i = 0;
  int j = 0;
 jumptarget: while (true) {
    foo();
    i++;
    j++;
    if (cond > 5)
      goto jumptarget;
  }
  return i + j;
}

// CHECK-LABEL: @_Z15switch_sidewaysi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: sw.bb:                                            ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb1:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb2:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: jumptarget:                                       ; preds = %sw.bb2, %sw.bb1
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.default:                                       ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]

// Convergence is well-defined for a jump within a switch statement.
__device__ int switch_sideways(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 20:
    foo();
    i += 2;
    goto jumptarget;
    break;
  case 30: {
    foo();
    i += 3;
  jumptarget:
    foo();
    i += 4;
    break;
  }
  default:
    foo();
    break;
  }
  return i;
}

// CHECK-LABEL: @_Z16switch_backwardsi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: sw.bb:                                            ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb1:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: jumptarget:                                       ; preds = %sw.bb3, %sw.bb1
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb3:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.default:                                       ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]

// Convergence is declared to be implementation-defined for a backward jump
// within a switch statement. But we don't actually do anything if it does not
// create a cycle. So it's perfectly legal that the convergent operations below
// use the entry token and not their own anchor token.
__device__ int switch_backwards(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 30: {
    foo();
    i += 3;
  jumptarget:
    foo();
    i += 4;
    break;
  }
  case 20:
    foo();
    i += 2;
    goto jumptarget;
    break;
  default:
    foo();
    break;
  }
  return i;
}

// CHECK-LABEL: @_Z18switch_fallthroughi
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: sw.bb:                                            ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb1:                                           ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.bb2:                                           ; preds = %entry, %sw.bb1
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: sw.default:                                       ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]

// Convergence is well-defined with fallthrough.
__device__ int switch_fallthrough(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 20:
    foo();
    i += 2;
  case 30: {
    foo();
    i += 3;
    i += 4;
    break;
  }
  default:
    foo();
    break;
  }
  return i;
}

// CHECK-LABEL: @_Z19forward_conditionali
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: if.then:                                          ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: jumptarget:                                       ; preds = %if.else, %if.then
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]

// Convergence is well-defined for forward jumps from the then-part to the
// else-part of a conditional statement.
__device__ int forward_conditional(int cond) {
  int i = 0;

  if (cond > 0) {
    foo();
    i++;
    goto jumptarget;
  } else {
  jumptarget:
    foo();
    i++;
  }

  return i;
}

// CHECK-LABEL: @_Z21backwards_conditionali
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: if.then:                                          ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: if.else:                                          ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]

// Convergence is declared to be implementation-defined for a backwards jump
// from the else-part to the then-part of a conditional statement. But just like
// the backwards jump in a switch, we don't actually do anything about it if it
// does not produce a cycle. Hence the convergent operations here use the entry
// token rather than their own anchor token.
__device__ int backwards_conditional(int cond) {
  int i = 0;

  foo();
  if (cond > 0) {
    foo();
  jumptarget:
    i++;
  } else {
    foo();
    i++;
    goto jumptarget;
  }

  return i;
}

// CHECK-LABEL: @_Z12duffs_devicei
// CHECK: entry:
// CHECK:   [[TOK0:%[0-9]+]] = call token @llvm.experimental.convergence.entry()
// CHECK: sw.bb:                                            ; preds = %entry
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK0]]) ]
// CHECK: do.body:                                          ; preds = %do.cond, %sw.bb
// CHECK:   [[TOK3:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK3]]) ]
// CHECK: sw.bb1:                                           ; preds = %entry, %do.body
// CHECK:   [[TOK5:%[0-9]+]] = call token @llvm.experimental.convergence.anchor()
// CHECK:   call void @_Z3foov() #3 [ "convergencectrl"(token [[TOK5]]) ]

// Note that the convergent operation in case 0 outside the do-while loop has
// well-defined convergence, and must use the entry token. The other operations
// must use anchor tokens.
__device__ int duffs_device(int count) {
  int i = 0;
  int n = (count + 7) / 8;
    switch (count % 8) {
    case 0:
      foo();
      do { i++;
        foo();
    case 7:      i++;
      foo();
    case 6:      i++;
    case 5:      i++;
    case 4:      i++;
    case 3:      i++;
    case 2:      i++;
    case 1:      i++;
            } while (--n > 0);
    }
    return i;
}
