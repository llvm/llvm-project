// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fclangir -emit-cir -clangir-disable-passes -Wno-return-type %s -o %t.cir
// RUN: cir-opt %t.cir -cir-fall-off-analysis -o /dev/null 2>&1 | FileCheck %s

// At -O1 the fall-off terminator is cir.unreachable rather than cir.trap.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -O1 -fclangir -emit-cir -clangir-disable-passes -Wno-return-type %s -o %t-O1.cir
// RUN: cir-opt %t-O1.cir -cir-fall-off-analysis -o /dev/null 2>&1 | FileCheck %s

// The warning annotations are what Sema's CFG-based check reports
// for the same code, so both classifications are pinned to each other.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -fsyntax-only -Wreturn-type -verify %s

[[noreturn]] void die();
int g();

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: always
int always(int x) {
  x++;
} // expected-warning {{non-void function does not return a value}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int maybe(int x) {
  if (x)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int never(int x) {
  if (x)
    return 1;
  return 0;
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int noreturn_call(int x) {
  if (x)
    return 1;
  die();
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int only_noreturn_call() {
  die();
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int user_unreachable(int x) {
  if (x)
    return 1;
  __builtin_unreachable();
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int user_trap(int x) {
  if (x)
    return 1;
  __builtin_trap();
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int noreturn_on_some_path(int x) {
  if (x)
    die();
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int constant_condition() {
  if (true)
    return 1;
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int infinite_loop() {
  while (true) {
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: always
int infinite_loop_with_break(int x) {
  while (true) {
    if (x)
      break;
  }
} // expected-warning {{non-void function does not return a value}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int for_ever(int x) {
  for (;;) {
    if (x)
      return 1;
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int while_may_not_run(int x) {
  while (x)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int do_while_runs_once(int x) {
  do {
    return 1;
  } while (x);
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int continue_reaches_exit(int x) {
  for (int i = 0; i < x; ++i) {
    if (i == 3)
      continue;
    return i;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int switch_with_default(int x) {
  switch (x) {
  case 0:
    return 0;
  default:
    return 1;
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int switch_without_default(int x) {
  switch (x) {
  case 0:
    return 0;
  case 1:
    return 1;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

enum E { A, B };

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int switch_covers_enum(E e) {
  switch (e) {
  case A:
    return 0;
  case B:
    return 1;
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int switch_break(int x) {
  switch (x) {
  case 0:
    break;
  default:
    return 1;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// A switch dispatches to cases nested anywhere in its body, even after dead
// code.
// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int switch_nested_case_after_dead_code(int x) {
  switch (x) {
    return 0;
    {
    case 1:
      return 1;
    }
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int switch_nested_default_after_dead_code(int x) {
  switch (x) {
    return 5;
    {
    default:
      return 1;
    }
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int switch_case_in_loop(int x, int n) {
  switch (x) {
  case 0:
    do {
      n--;
    case 1:
      n--;
    } while (n > 0);
    return n;
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int switch_fallthrough(int x) {
  switch (x) {
  case 0:
    x++;
    [[fallthrough]];
  default:
    return x;
  }
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int throw_on_some_path(int x) {
  if (x)
    return 1;
  throw 0;
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int only_throw() {
  throw 1;
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int catch_falls_off() {
  try {
    return g();
  } catch (...) {
  }
} // expected-warning {{non-void function does not return a value in all control paths}}

struct NoReturnDtor {
  [[noreturn]] ~NoReturnDtor();
};

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int noreturn_destructor() {
  NoReturnDtor d;
}

struct Dtor {
  ~Dtor();
};

// The body of a function with a destructor to run is a cleanup scope.
// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int cleanup_scope_falls_off(int x) {
  Dtor d;
  if (x)
    return 1;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int cleanup_scope_returns() {
  Dtor d;
  return 1;
  g();
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int goto_to_return(int x) {
  if (x)
    goto out;
  return 0;
out:
  return 1;
}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int goto_to_end(int x) {
  if (x)
    goto out;
  return 0;
out:;
} // expected-warning {{non-void function does not return a value in all control paths}}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: always
void void_function() {}

// CHECK-DAG: fall-off.cpp:[[@LINE+1]]:1: remark: fall-off: never
int use_lambda(int x) {
  // CHECK-DAG: fall-off.cpp:[[@LINE+1]]:{{[0-9]+}}: remark: fall-off: maybe
  auto l = [](int y) -> int {
    if (y)
      return 1;
  }; // expected-warning {{non-void lambda does not return a value in all control paths}}
  return l(x);
}
