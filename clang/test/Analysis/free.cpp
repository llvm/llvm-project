// RUN: %clang_analyze_cc1 -fblocks -verify %s -analyzer-store=region \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Malloc
//
// RUN: %clang_analyze_cc1 -fblocks -verify %s -analyzer-store=region \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.Malloc \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:Optimistic=true
namespace std {
  using size_t = decltype(sizeof(int));
  void free(void *);
}

extern "C" void free(void *);
extern "C" void *alloca(std::size_t);

void t1a () {
  int a[] = { 1 };
  free(a);
  // expected-warning@-1{{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t1b () {
  int a[] = { 1 };
  std::free(a);
  // expected-warning@-1{{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'a'}}
}

void t2a () {
  int a = 1;
  free(&a);
  // expected-warning@-1{{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t2b () {
  int a = 1;
  std::free(&a);
  // expected-warning@-1{{Argument to free() is the address of the local variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'a'}}
}

void t3a () {
  static int a[] = { 1 };
  free(a);
  // expected-warning@-1{{Argument to free() is the address of the static variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t3b () {
  static int a[] = { 1 };
  std::free(a);
  // expected-warning@-1{{Argument to free() is the address of the static variable 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'a'}}
}

void t4a (char *x) {
  free(x); // no-warning
}

void t4b (char *x) {
  std::free(x); // no-warning
}

void t5a () {
  extern char *ptr();
  free(ptr()); // no-warning
}

void t5b () {
  extern char *ptr();
  std::free(ptr()); // no-warning
}

void t6a () {
  free((void*)1000);
  // expected-warning@-1{{Argument to free() is a constant address (1000), which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object '(void *)1000'}}
}

void t6b () {
  std::free((void*)1000);
  // expected-warning@-1{{Argument to free() is a constant address (1000), which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object '(void *)1000'}}
}

void t7a (char **x) {
  free(*x); // no-warning
}

void t7b (char **x) {
  std::free(*x); // no-warning
}

void t8a (char **x) {
  // ugh
  free((*x)+8); // no-warning
}

void t8b (char **x) {
  // ugh
  std::free((*x)+8); // no-warning
}

void t9a () {
label:
  free(&&label);
  // expected-warning@-1{{Argument to free() is the address of the label 'label', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'label'}}
}

void t9b () {
label:
  std::free(&&label);
  // expected-warning@-1{{Argument to free() is the address of the label 'label', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'label'}}
}

void t10a () {
  free((void*)&t10a);
  // expected-warning@-1{{Argument to free() is the address of the function 't10a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 't10a'}}
}

void t10b () {
  std::free((void*)&t10b);
  // expected-warning@-1{{Argument to free() is the address of the function 't10b', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 't10b'}}
}

void t11a () {
  char *p = (char*)alloca(2);
  free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t11b () {
  char *p = (char*)alloca(2);
  std::free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t12a () {
  char *p = (char*)__builtin_alloca(2);
  free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t12b () {
  char *p = (char*)__builtin_alloca(2);
  std::free(p); // expected-warning {{Memory allocated by alloca() should not be deallocated}}
}

void t13a () {
  free(^{return;});
  // expected-warning@-1{{Argument to free() is a block, which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object: block expression}}
}

void t13b () {
  std::free(^{return;});
  // expected-warning@-1{{Argument to free() is a block, which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object: block expression}}
}

void t14a () {
  free((void *)+[]{ return; });
  // expected-warning@-1{{Argument to free() is the address of the function '__invoke', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object: lambda-to-function-pointer conversion}}
}

void t14b () {
  std::free((void *)+[]{ return; });
  // expected-warning@-1{{Argument to free() is the address of the function '__invoke', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object: lambda-to-function-pointer conversion}}
}

void t15a (char a) {
  free(&a);
  // expected-warning@-1{{Argument to free() is the address of the parameter 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'a'}}
}

void t15b (char a) {
  std::free(&a);
  // expected-warning@-1{{Argument to free() is the address of the parameter 'a', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'a'}}
}

static int someGlobal[2];
void t16a () {
  free(someGlobal);
  // expected-warning@-1{{Argument to free() is the address of the global variable 'someGlobal', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call free on non-heap object 'someGlobal'}}
}

void t16b () {
  std::free(someGlobal);
  // expected-warning@-1{{Argument to free() is the address of the global variable 'someGlobal', which is not memory allocated by malloc()}}
  // expected-warning@-2{{attempt to call std::free on non-heap object 'someGlobal'}}
}

void t17a (char **x, int offset) {
  // Unknown value
  free(x[offset]); // no-warning
}

void t17b (char **x, int offset) {
  // Unknown value
  std::free(x[offset]); // no-warning
}

struct S {
  const char* p;
};

void t18_C_style_C_style_free (S s) {
  free((void*)(unsigned long long)s.p); // no warning
}

void t18_C_style_C_style_std_free (S s) {
  std::free((void*)(unsigned long long)s.p); // no warning
}

void t18_C_style_reinterpret_free (S s) {
  free((void*)reinterpret_cast<unsigned long long>(s.p)); // no warning
}

void t18_C_style_reinterpret_std_free (S s) {
  std::free((void*)reinterpret_cast<unsigned long long>(s.p)); // no warning
}

void t18_reinterpret_C_style_free (S s) {
  free(reinterpret_cast<void*>((unsigned long long)(s.p))); // no warning
}

void t18_reinterpret_C_style_std_free (S s) {
  std::free(reinterpret_cast<void*>((unsigned long long)(s.p))); // no warning
}

void t18_reinterpret_reinterpret_free (S s) {
  free(reinterpret_cast<void*>(reinterpret_cast<unsigned long long>(s.p))); // no warning
}

void t18_reinterpret_reinterpret_std_free (S s) {
  std::free(reinterpret_cast<void*>(reinterpret_cast<unsigned long long>(s.p))); // no warning
}
