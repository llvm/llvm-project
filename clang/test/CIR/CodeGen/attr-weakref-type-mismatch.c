// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify -DVAR %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify -DFN %s -o %t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -verify -DTENTATIVE %s -o %t.cir

// A weakref whose target name was already declared with a different type (or
// kind) can't be reused as-is, since CIR symbols are typed.

#ifdef VAR
extern long tgt_var;
long use_var(void) { return tgt_var; }
// expected-error@+1 {{ClangIR code gen Not Yet Implemented: weakref target already declared with a different type}}
static int r_var __attribute__((weakref("tgt_var")));
int *get_r_var(void) { return &r_var; }
#endif

#ifdef FN
int tgt_fn(void);
int use_fn(void) { return tgt_fn(); }
// expected-error@+1 {{ClangIR code gen Not Yet Implemented: weakref target already declared with a different type}}
static void r_fn(void) __attribute__((weakref("tgt_fn")));
void call_r_fn(void) { r_fn(); }
#endif

// A function weakref whose target is later defined as a variable.
#ifdef TENTATIVE
static void r_fn2(void) __attribute__((weakref("gv")));
// expected-error@+1 {{ClangIR code gen Not Yet Implemented: emitTentativeDefinition: name already used by a non-global}}
int gv;
void call_r_fn2(void) { r_fn2(); }
#endif
