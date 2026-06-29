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
#endif // __cplusplus

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

[[noreturn]] void direct_noreturn(void);
// FIXME: the cast should not be necessary.
void (*indirect_noreturn)(void) __attribute__((noreturn)) = (__typeof__(indirect_noreturn)) direct_noreturn;
void returns_okay();

__attribute__((const)) int noreturn_test1(void) {
  returns_okay();
  return 12;
}

__attribute__((const)) int noreturn_test2(void) { // expected-note {{function declared 'const' here}}
  direct_noreturn(); // expected-warning {{alling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  return 12;
}

__attribute__((const)) int noreturn_test3(void) { // expected-note {{function declared 'const' here}}
  indirect_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  return 12;
}

__attribute__((const)) int noreturn_test4(void) { // expected-note {{function declared 'const' here}}
  // This should not be diagnosed.
  (void)sizeof((direct_noreturn(), 1));

#ifdef __cplusplus
  if constexpr(false) {
	// This should not be diagnosed.
    direct_noreturn();
  }
#endif // __cplusplus

  if (0) {
	// FIXME: it would be better if this was not diagnosed because it is
	// statically known to be unreachable.
    direct_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  }

  return 12;
}

__attribute__((pure)) int noreturn_test5(int x) { // expected-note {{function declared 'pure' here}}
  if (x)
    direct_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'pure' attribute is undefined behavior}}
  return 12;
}

// FIXME: should this be diagnosed because of the noreturn call?
[[gnu::pure]] int noreturn_test6(int array[(direct_noreturn(), 1)]);

#ifdef __cplusplus

template <typename Ty>
int noreturn_test7(void) {
  direct_noreturn(); // okay
  return 12;
}

template <>
__attribute__((const)) int noreturn_test7<int>() { // expected-note {{function declared 'const' here}}
  direct_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  return 12;
}

template <typename Ty>
__attribute__((pure)) int noreturn_test8() { // expected-note {{function declared 'pure' here}}
  // Diagnosed even though noreturn_test8 is not instantiated
  direct_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'pure' attribute is undefined behavior}}
  return 12;
}

template <typename T>
[[gnu::pure]] int noreturn_test9() { // expected-note {{function declared 'pure' here}}
  T::nrcall(); // expected-warning {{calling a 'noreturn' function from a function with the 'pure' attribute is undefined behavior}}
  return 12;
}

struct S {
  [[noreturn]] static void nrcall();
  [[noreturn]] void mem_nrcall();

  void (*indirect_mem_noreturn)(void) __attribute__((noreturn));
};

void instantiate() {
  (void)noreturn_test9<S>(); // expected-note {{in instantiation of function template specialization 'noreturn_test9<S>' requested here}}
}

[[gnu::const]] int memfn() {   // expected-note 2 {{function declared 'const' here}}
  S{}.mem_nrcall();            // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  S{}.indirect_mem_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
}

__attribute__((pure)) int noreturn_test10() {
  (void)[] {
	// This should not be diagnosed, it's not called within the pure function.
    direct_noreturn();
  };
  
  (void)[]() __attribute__((const)) { // expected-note {{function declared 'const' here}}
	direct_noreturn(); // expected-warning {{calling a 'noreturn' function from a function with the 'const' attribute is undefined behavior}}
  };
  
  return 12;
}
#endif // __cplusplus
