// RUN: %clang_cc1 -verify=default-expected -ftyped-memory-operations -fsyntax-only -nostdsysteminc %s
// RUN: %clang_cc1 -verify -ftyped-memory-operations -Wtyped-memory-inference-failure -fsyntax-only -nostdsysteminc %s
// RUN: %clang_cc1 -verify=disabled-expected -fno-typed-memory-operations -Wtyped-memory-inference-failure -fsyntax-only -nostdsysteminc %s

// default-expected-no-diagnostics
// disabled-expected-no-diagnostics
#define _TYPED(rewrite_target, type_param_pos) __attribute__((typed_memory_operation(rewrite_target, type_param_pos)))

void *typed_malloc(__SIZE_TYPE__ size, unsigned long long);
void *malloc(__SIZE_TYPE__ size) _TYPED(typed_malloc, 1);

template<typename T, __SIZE_TYPE__ S>
void f() {
  __SIZE_TYPE__ sz;
  int *p;
  void *v;

  malloc(sizeof(int));
  malloc(sizeof(T));
  malloc(S); // #f_malloc_S

  p = (int *)malloc(sz);
  
  p = (int *)malloc(S);
  malloc(sz);// #f_malloc_sz
  v = (void *)malloc(S); // #f_voidptr_malloc_S
  
  p = reinterpret_cast<int *>(malloc(sz));
  p = static_cast<int *>(malloc(sz));
  
  p = reinterpret_cast<int *>(malloc(sizeof(T)));
  p = static_cast<int *>(malloc(sizeof(T)));
  
  p = reinterpret_cast<int *>(malloc(S));
  p = static_cast<int *>(malloc(S));
}

template<__SIZE_TYPE__ S>
void i() {
  malloc(S); // #i_malloc_S
  void *v = (void *)malloc(S);  // #i_void_malloc
}

void g() {
  f<int, sizeof(int)>(); // #f_call
  // expected-note@#f_call {{in instantiation of function template specialization 'f<int, 4UL>' requested here}}
  // expected-warning@#f_malloc_S{{could not infer allocation type in call to 'malloc'}}
  // expected-warning@#f_malloc_sz {{could not infer allocation type in call to 'malloc'}}
  // expected-warning@#f_voidptr_malloc_S{{could not infer allocation type in call to 'malloc'}}

  i<sizeof(int)>();
  // expected-note@-1 {{in instantiation of function template specialization 'i<4UL>' requested here}}
  // expected-warning@#i_malloc_S {{could not infer allocation type in call to 'malloc'}}
  // expected-warning@#i_void_malloc {{could not infer allocation type in call to 'malloc'}}
}

void h() {
  __SIZE_TYPE__ sz;
  int *p;
  void *v;

  malloc(sizeof(int));

  p = (int *)malloc(sz);

  malloc(sz);
  // expected-warning@-1{{could not infer allocation type in call to 'malloc'}}
  
  p = reinterpret_cast<int *>(malloc(sizeof(int)));
  
  p = reinterpret_cast<int *>(malloc(sz));
  p = static_cast<int *>(malloc(sz));
}
