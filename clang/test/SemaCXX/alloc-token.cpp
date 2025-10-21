// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typehash -DMODE_TYPEHASH

#if !__has_builtin(__builtin_infer_alloc_token)
#error "missing __builtin_infer_alloc_token"
#endif

template <typename T = void>
void template_test() {
  __builtin_infer_alloc_token(T()); // no error if not instantiated
}

template <typename T>
void negative_template_test() {
  __builtin_infer_alloc_token(T()); // expected-error {{argument may not have 'void' type}}
}

void negative_tests() {
  __builtin_infer_alloc_token(); // expected-error {{too few arguments to function call}}
  __builtin_infer_alloc_token((void)0); // expected-error {{argument may not have 'void' type}}
  negative_template_test<void>(); // expected-note {{in instantiation of function template specialization 'negative_template_test<void>' requested here}}
}
