// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// Regression test for a CIRGenModule bug where replaceGlobal would erase the
// cir.global tracked by CIRGenModule::lastGlobalOp without updating that
// pointer, leaving it dangling.

struct State {
  char buf[33];
  int b;
};

static const struct State g_state = { "x", 0 };

static void helper(struct State *s) {
  *s = g_state;
}

static const char *names[] = { "n1", "n2" };

extern void use_str(const char *);

void api(struct State *s, int idx) {
  helper(s);
  use_str(names[idx]);
}

// CIR: cir.global {{.*}} @names = #cir.const_array<[#cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.global_view<@".str.1"> : !cir.ptr<!s8i>]>
// CIR: cir.global {{.*}} @g_state =
// CIR: cir.global {{.*}} @".str" = #cir.const_array<"n1" :
// CIR: cir.global {{.*}} @".str.1" = #cir.const_array<"n2" :

// LLVM: @names = internal global [2 x ptr] [ptr @[[STR0:\.str(\.[0-9]+)?]], ptr @[[STR1:\.str(\.[0-9]+)?]]]
// LLVM: @g_state = internal constant
// LLVM: @[[STR0]] = {{.*}}constant [3 x i8] c"n1\00"
// LLVM: @[[STR1]] = {{.*}}constant [3 x i8] c"n2\00"
