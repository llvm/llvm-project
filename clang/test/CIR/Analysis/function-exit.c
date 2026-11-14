// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -clangir-disable-passes -Wno-return-type %s -o %t.cir
// RUN: cir-opt %t.cir -cir-function-exit-analysis -o /dev/null 2>&1 | FileCheck %s
// RUN: cir-opt %t.cir -cir-function-exit-analysis --mlir-disable-threading -o /dev/null 2>&1 | FileCheck %s
// RUN: cir-opt %t.cir -cir-function-exit-analysis -o /dev/null 2>&1 | FileCheck %s --check-prefix=ALIAS

// The warning annotations are what Sema's CFG-based check reports for the
// same code, so both classifications are pinned to each other.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -Wreturn-type -verify %s

_Noreturn void stop(void);

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int may_reach_end(int x) {
  if (x)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int while_one(int x) {
  while (1) {
    if (x)
      return 1;
  }
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: must_reach_end
int do_while_zero(int x) {
  do {
    if (x)
      break;
  } while (0);
} // expected-warning {{non-void function does not return a value}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: no_return
int only_noreturn_call(void) {
  stop();
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: no_return
int goto_loop(void) {
loop:
  goto loop;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int computed_goto(int x) {
  void *target = x ? &&one : &&two;
  goto *target;
one:
  return 1;
two:
  return 2;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int computed_goto_to_end(int x) {
  void *target = x ? &&one : &&two;
  goto *target;
one:
  return 1;
two:;
} // expected-warning {{non-void function does not return a value in all control paths}}

// Like in the CFG, a label whose address is taken is reached only through a
// reachable indirect goto, including addresses taken in static initializers.
// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int address_taken_without_indirect_goto(void) {
  void *p = &&l;
  (void)p;
  return 0;
l:;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int dead_indirect_goto(void) {
  return 0;
  void *p = &&l;
  goto *p;
l:;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int static_table_indirect_goto(int i) {
  static void *table[] = {&&one, &&two};
  goto *table[i];
one:
  return 1;
two:;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int dead_goto(int x) {
  while (0) {
    goto out;
  }
  return 1;
out:;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int constant_or(int x) {
  if (x == 1)
    return 1;
  else if (x == 2 || 1)
    return 2;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int nonconstant_or(int x) {
  if (x == 1 || x == 2)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int assume_false(int x) {
  if (x)
    return 1;
  __builtin_assume(0);
}

// A goto into the tail of the last scope reaches the end past its return.
// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int goto_into_trailing_scope(int x) {
  if (x)
    goto inside;
  {
    return 1;
  inside:;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// A goto into the last statement reaches its end past the returns before.
// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int goto_into_trailing_switch(int x) {
  if (x > 1)
    goto inside;
  switch (x) {
  case 0:
    return 0;
  default:
    return 1;
  inside:;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int goto_past_noreturn_in_trailing_if(int x) {
  if (x > 1)
    goto inside;
  if (x) {
    stop();
    {
    inside:;
    }
  } else {
    return 0;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: return_only
int trailing_if_else_returns(int x) {
  if (x)
    return 1;
  else
    return 2;
}

// CHECK: function-exit.c:[[@LINE+1]]:1: remark: function exit: may_reach_end
int trailing_loop_breaks(int x) {
  while (1) {
    if (x)
      return 1;
    if (x < 0)
      break;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// An alias has no body of its own to classify.
// ALIAS-NOT: function-exit.c:[[@LINE+1]]:{{.*}}remark
int alias_of_may_reach_end(int x) __attribute__((alias("may_reach_end")));
