// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

// The attributes apply to function declarations, nothing else.
__attribute__((const)) int func1(void);
__attribute__((pure)) int func2(void);

[[gnu::const]] int func3(int x) { return 12; }
[[gnu::pure]] int func4(int x) { return 12; }

#ifdef __cplusplus
struct CppTest {
  // They are fine on member functions.
  __attribute__((const)) int func();
  [[gnu::pure]] int other_func();
  
  int another();
  
  // Constructors and destructors are not allowed though because they
  // notionally return void.
  [[__gnu__::__const__]] CppTest(); // expected-warning {{'const' attribute on function returning 'void'; attribute ignored}}
  __attribute__((pure)) ~CppTest(); // expected-warning {{'pure' attribute on function returning 'void'; attribute ignored}}
};

// Including out-of-line member functions.
__attribute__((pure)) int CppTest::another() { return 12; }

// They also work on function templates.
template <typename Ty>
__attribute__((const)) int temp_func1(Ty);

// And specializations, too.
// FIXME: this should be diagnosed because it ends up with both the const and pure attributes.
template <>
[[gnu::pure]] int temp_func1<int>(int) { return 12; }
#endif

// They do not apply to types, including function pointer types.
int (*fp1)(void) [[gnu::const]]; // expected-warning {{attribute 'gnu::const' ignored, because it cannot be applied to a type}}
int (*fp2)(void) [[gnu::pure]];  // expected-warning {{attribute 'gnu::pure' ignored, because it cannot be applied to a type}}

struct __attribute__((const)) S1 { // expected-warning {{'const' attribute only applies to functions}}
  int x;
};

struct __attribute__((pure)) S2 { // expected-warning {{'pure' attribute only applies to functions}}
  int x;
};

// Or variables, etc.
__attribute__((const)) int variable;   // expected-warning {{'const' attribute only applies to functions}}
__attribute__((pure)) typedef int foo; // expected-warning {{'pure' attribute only applies to functions}}

// The function they apply to should return non-void.
__attribute__((const)) void func5(int x); // expected-warning {{'const' attribute on function returning 'void'; attribute ignored}}
__attribute__((pure)) void func6(int x);  // expected-warning {{'pure' attribute on function returning 'void'; attribute ignored}}

// The attributes cannot be used together.
__attribute__((const, pure)) int func7(void); // expected-warning {{'const' attribute imposes more restrictions; 'pure' attribute ignored}}

// FIXME: this should also be diagnosed the same as func7.
__attribute__((pure)) int func8(void);
[[gnu::const]] int func8(void) {
  return 12;
}

