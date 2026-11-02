// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typefunchash
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typefunchashpointersplit
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typefunchash -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typefunchashpointersplit -fexperimental-new-constant-interpreter

// The token ID depends on the function containing the allocation, which is only
// known to the AllocToken pass: the builtin is not a constant expression.
static_assert(!__builtin_constant_p(__builtin_infer_alloc_token(sizeof(int))));

void test() {
  constexpr auto token = __builtin_infer_alloc_token(sizeof(int)); // expected-error {{must be initialized by a constant expression}} \
                                                                   // expected-note {{alloc token mode not supported in constexpr}}
  auto runtime_token = __builtin_infer_alloc_token(sizeof(int));
}
