// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fobjc-runtime=macosx-10.14 -fobjc-arc -emit-llvm -o /dev/null %s -verify

// Test for - https://github.com/llvm/llvm-project/issues/222528

@class incompatible;

static incompatible *g;

void integer(incompatible *o) {
  __sync_bool_compare_and_swap(&g, 0, o); // expected-error {{cannot perform atomic operation on a pointer to type 'incompatible *__strong': type has non-trivial ownership}} \
                                          // expected-error {{cannot compile this scalar expression yet}}
}
