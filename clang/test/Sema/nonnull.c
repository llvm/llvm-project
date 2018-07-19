// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://9584012

typedef struct {
	char *str;
} Class;

typedef union {
	Class *object;
} Instance __attribute__((transparent_union));

__attribute__((nonnull(1))) void Class_init(Instance this, char *str) {
	this.object->str = str;
}

int main(void) {
	Class *obj;
	Class_init(0, "Hello World"); // expected-warning {{null passed to a callee that requires a non-null argument}}
	Class_init(obj, "Hello World");
}

void foo(const char *str) __attribute__((nonnull("foo"))); // expected-error{{'nonnull' attribute requires parameter 1 to be an integer constant}}
void bar(int i) __attribute__((nonnull(1))); // expected-warning {{'nonnull' attribute only applies to pointer arguments}} expected-warning {{'nonnull' attribute applied to function with no pointer arguments}}

void baz(__attribute__((nonnull)) const char *str);
void baz2(__attribute__((nonnull(1))) const char *str); // expected-warning {{'nonnull' attribute when used on parameters takes no arguments}}
void baz3(__attribute__((nonnull)) int x); // expected-warning {{'nonnull' attribute only applies to pointer arguments}}

void test_baz() {
  baz(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
  baz2(0); // no-warning
  baz3(0); // no-warning
}

void test_void_returns_nonnull(void) __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to return values that are pointers}}
int test_int_returns_nonnull(void) __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to return values that are pointers}}
void *test_ptr_returns_nonnull(void) __attribute__((returns_nonnull)); // no-warning

int i __attribute__((nonnull)); // expected-warning {{'nonnull' attribute only applies to functions, methods, and parameters}}
int j __attribute__((returns_nonnull)); // expected-warning {{'returns_nonnull' attribute only applies to Objective-C methods and functions}}
void *test_no_fn_proto() __attribute__((returns_nonnull)); // no-warning
void *test_with_fn_proto(void) __attribute__((returns_nonnull)); // no-warning

__attribute__((returns_nonnull))
void *test_bad_returns_null(void) {
  return 0; // expected-warning {{null returned from function that requires a non-null return value}}
}

void PR18795(int (*g)(const char *h, ...) __attribute__((nonnull(1))) __attribute__((nonnull))) {
  g(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}
void PR18795_helper() {
  PR18795(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

void vararg1(int n, ...) __attribute__((nonnull(2)));
void vararg1_test() {
  vararg1(0);
  vararg1(1, (void*)0); // expected-warning{{null passed}}
  vararg1(2, (void*)0, (void*)0); // expected-warning{{null passed}}
  vararg1(2, (void*)&vararg1, (void*)0);
}

void vararg2(int n, ...) __attribute__((nonnull, nonnull, nonnull));
void vararg2_test() {
  vararg2(0);
  vararg2(1, (void*)0); // expected-warning{{null passed}}
  vararg2(2, (void*)0, (void*)0); // expected-warning 2{{null passed}}
}

void vararg3(int n, ...) __attribute__((nonnull, nonnull(2), nonnull(3)));
void vararg3_test() {
  vararg3(0);
  vararg3(1, (void*)0); // expected-warning{{null passed}}
  vararg3(2, (void*)0, (void*)0); // expected-warning 2{{null passed}}
}

void redecl(void *, void *);
void redecl(void *, void *) __attribute__((nonnull(1)));
void redecl(void *, void *) __attribute__((nonnull(2)));
void redecl(void *, void *);
void redecl_test(void *p) {
  redecl(p, 0); // expected-warning{{null passed}}
  redecl(0, p); // expected-warning{{null passed}}
}

// rdar://18712242
#define NULL (void*)0
__attribute__((__nonnull__))  // expected-note 2{{declared 'nonnull' here}}
int evil_nonnull_func(int* pointer, void * pv)
{
   if (pointer == NULL) {  // expected-warning {{comparison of nonnull parameter 'pointer' equal to a null pointer is 'false' on first encounter}}
     return 0;
   } else {
     return *pointer;
   } 

   pointer = pv;
   if (!pointer)
     return 0;
   else
     return *pointer;

   if (pv == NULL) {} // expected-warning {{comparison of nonnull parameter 'pv' equal to a null pointer is 'false' on first encounter}}
}

void set_param_to_null(int**);
int another_evil_nonnull_func(int* pointer, char ch, void * pv) __attribute__((nonnull(1, 3)));  // expected-note 2{{declared 'nonnull' here}}
int another_evil_nonnull_func(int* pointer, char ch, void * pv) {
   if (pointer == NULL) { // expected-warning {{comparison of nonnull parameter 'pointer' equal to a null pointer is 'false' on first encounter}}
     return 0;
   } else {
     return *pointer;
   } 

   set_param_to_null(&pointer);
   if (!pointer)
     return 0;
   else
     return *pointer;

   if (pv == NULL) {} // expected-warning {{comparison of nonnull parameter 'pv' equal to a null pointer is 'false' on first encounter}}
}

extern void *returns_null(void**);
extern void FOO();
extern void FEE();

extern void *pv;
__attribute__((__nonnull__))  // expected-note {{declared 'nonnull' here}}
void yet_another_evil_nonnull_func(int* pointer)
{
 while (pv) {
   // This comparison will not be optimized away.
   if (pointer) {  // expected-warning {{nonnull parameter 'pointer' will evaluate to 'true' on first encounter}}
     FOO();
   } else {
     FEE();
   } 
   pointer = returns_null(&pv);
 }
}

void pr21668_1(__attribute__((nonnull)) const char *p, const char *s) { // expected-note {{declared 'nonnull' here}}
  if (p) // expected-warning {{nonnull parameter 'p' will evaluate to 'true' on first encounter}}
    ;
  if (s) // No warning
    ;
}

void pr21668_2(__attribute__((nonnull)) const char *p) {
  p = 0;
  if (p) // No warning
    ;
}

__attribute__((returns_nonnull)) void *returns_nonnull_whee();  // expected-note 6{{declared 'returns_nonnull' here}}

void returns_nonnull_warning_tests() {
  if (returns_nonnull_whee() == NULL) {} // expected-warning {{comparison of nonnull function call 'returns_nonnull_whee()' equal to a null pointer is 'false' on first encounter}}

  if (returns_nonnull_whee() != NULL) {} // expected-warning {{comparison of nonnull function call 'returns_nonnull_whee()' not equal to a null pointer is 'true' on first encounter}}

  if (returns_nonnull_whee()) {} // expected-warning {{nonnull function call 'returns_nonnull_whee()' will evaluate to 'true' on first encounter}}
  if (!returns_nonnull_whee()) {} // expected-warning {{nonnull function call 'returns_nonnull_whee()' will evaluate to 'true' on first encounter}}

  int and_again = !returns_nonnull_whee(); // expected-warning {{nonnull function call 'returns_nonnull_whee()' will evaluate to 'true' on first encounter}}
  and_again = !returns_nonnull_whee(); // expected-warning {{nonnull function call 'returns_nonnull_whee()' will evaluate to 'true' on first encounter}}
}

void pr30828(char *p __attribute__((nonnull)));
void pr30828(char *p) {}

void call_pr30828() {
  pr30828(0); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
