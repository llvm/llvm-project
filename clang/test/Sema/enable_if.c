// RUN: %clang_cc1 %s -verify
// RUN: %clang_cc1 %s -DCODEGEN -emit-llvm -o - | FileCheck %s

#define O_CREAT 0x100
typedef int mode_t;
typedef unsigned long size_t;

const int TRUE = 1;

int open(const char *pathname, int flags) __attribute__((enable_if(!(flags & O_CREAT), "must specify mode when using O_CREAT"))) __attribute__((overloadable));  // expected-note{{candidate disabled: must specify mode when using O_CREAT}}
int open(const char *pathname, int flags, mode_t mode) __attribute__((overloadable));  // expected-note{{candidate function not viable: requires 3 arguments, but 2 were provided}}

void test1() {
#ifndef CODEGEN
  open("path", O_CREAT);  // expected-error{{no matching function for call to 'open'}}
#endif
  open("path", O_CREAT, 0660);
  open("path", 0);
  open("path", 0, 0);
}

size_t __strnlen_chk(const char *s, size_t requested_amount, size_t s_len);

size_t strnlen(const char *s, size_t maxlen)  // expected-note{{candidate function}}
  __attribute__((overloadable))
  __asm__("strnlen_real1");

__attribute__((always_inline))
inline size_t strnlen(const char *s, size_t maxlen)  // expected-note{{candidate function}}
  __attribute__((overloadable))
  __attribute__((enable_if(__builtin_object_size(s, 0) != -1,
                           "chosen when target buffer size is known")))
{
  return __strnlen_chk(s, maxlen, __builtin_object_size(s, 0));
}

size_t strnlen(const char *s, size_t maxlen)  // expected-note{{candidate disabled: chosen when 'maxlen' is known to be less than or equal to the buffer size}}
  __attribute__((overloadable))
  __attribute__((enable_if(__builtin_object_size(s, 0) != -1,
                           "chosen when target buffer size is known")))
  __attribute__((enable_if(maxlen <= __builtin_object_size(s, 0),
                           "chosen when 'maxlen' is known to be less than or equal to the buffer size")))
  __asm__("strnlen_real2");

size_t strnlen(const char *s, size_t maxlen)  // expected-note{{candidate function has been explicitly made unavailable}}
  __attribute__((overloadable))
  __attribute__((enable_if(__builtin_object_size(s, 0) != -1,
                           "chosen when target buffer size is known")))
  __attribute__((enable_if(maxlen > __builtin_object_size(s, 0),
                           "chosen when 'maxlen' is larger than the buffer size")))
  __attribute__((unavailable("'maxlen' is larger than the buffer size")));

void test2(const char *s, int i) {
// CHECK: define {{.*}}void @test2
  const char c[123];
  strnlen(s, i);
// CHECK: call {{.*}}strnlen_real1
  strnlen(s, 999);
// CHECK: call {{.*}}strnlen_real1
  strnlen(c, 1);
// CHECK: call {{.*}}strnlen_real2
  strnlen(c, i);
// CHECK: call {{.*}}strnlen_chk
#ifndef CODEGEN
  strnlen(c, 999);  // expected-error{{call to unavailable function 'strnlen': 'maxlen' is larger than the buffer size}}
#endif
}

int isdigit(int c) __attribute__((overloadable));  // expected-note{{candidate function}}
int isdigit(int c) __attribute__((overloadable))  // expected-note{{candidate function has been explicitly made unavailable}}
  __attribute__((enable_if(c <= -1 || c > 255, "'c' must have the value of an unsigned char or EOF")))
  __attribute__((unavailable("'c' must have the value of an unsigned char or EOF")));

void test3(int c) {
  isdigit(c);
  isdigit(10);
#ifndef CODEGEN
  isdigit(-10);  // expected-error{{call to unavailable function 'isdigit': 'c' must have the value of an unsigned char or EOF}}
#endif
}

// Verify that the alternate spelling __enable_if__ works as well.
int isdigit2(int c) __attribute__((overloadable));  // expected-note{{candidate function}}
int isdigit2(int c) __attribute__((overloadable))  // expected-note{{candidate function has been explicitly made unavailable}}
  __attribute__((__enable_if__(c <= -1 || c > 255, "'c' must have the value of an unsigned char or EOF")))
  __attribute__((unavailable("'c' must have the value of an unsigned char or EOF")));

void test4(int c) {
  isdigit2(c);
  isdigit2(10);
#ifndef CODEGEN
  isdigit2(-10);  // expected-error{{call to unavailable function 'isdigit2': 'c' must have the value of an unsigned char or EOF}}
#endif
}

void test5() {
  int (*p1)(int) = &isdigit2;
  int (*p2)(int) = isdigit2;
  void *p3 = (void *)&isdigit2;
  void *p4 = (void *)isdigit2;
}

#ifndef CODEGEN
__attribute__((enable_if(n == 0, "chosen when 'n' is zero"))) void f1(int n); // expected-error{{use of undeclared identifier 'n'}}

int n __attribute__((enable_if(1, "always chosen"))); // expected-warning{{'enable_if' attribute only applies to functions}}

void f(int n) __attribute__((enable_if("chosen when 'n' is zero", n == 0)));  // expected-error{{'enable_if' attribute requires a string}}

void f(int n) __attribute__((enable_if()));  // expected-error{{'enable_if' attribute requires exactly 2 arguments}}

void f(int n) __attribute__((enable_if(unresolvedid, "chosen when 'unresolvedid' is non-zero")));  // expected-error{{use of undeclared identifier 'unresolvedid'}}

int global;
void f(int n) __attribute__((enable_if(global == 0, "chosen when 'global' is zero")));  // expected-error{{'enable_if' attribute expression never produces a constant expression}}  // expected-note{{subexpression not valid in a constant expression}}

const int cst = 7;
void return_cst(void) __attribute__((overloadable)) __attribute__((enable_if(cst == 7, "chosen when 'cst' is 7")));
void test_return_cst() { return_cst(); }

void f2(void) __attribute__((overloadable)) __attribute__((enable_if(1, "always chosen")));
void f2(void) __attribute__((overloadable)) __attribute__((enable_if(0, "never chosen")));
void f2(void) __attribute__((overloadable)) __attribute__((enable_if(TRUE, "always chosen #2")));
void test6() {
  void (*p1)(void) = &f2; // expected-error{{initializing 'void (*)(void)' with an expression of incompatible type '<overloaded function type>'}} expected-note@121{{candidate function}} expected-note@122{{candidate function made ineligible by enable_if}} expected-note@123{{candidate function}}
  void (*p2)(void) = f2; // expected-error{{initializing 'void (*)(void)' with an expression of incompatible type '<overloaded function type>'}} expected-note@121{{candidate function}} expected-note@122{{candidate function made ineligible by enable_if}} expected-note@123{{candidate function}}
  void *p3 = (void*)&f2; // expected-error{{address of overloaded function 'f2' is ambiguous}} expected-note@121{{candidate function}} expected-note@122{{candidate function made ineligible by enable_if}} expected-note@123{{candidate function}}
  void *p4 = (void*)f2; // expected-error{{address of overloaded function 'f2' is ambiguous}} expected-note@121{{candidate function}} expected-note@122{{candidate function made ineligible by enable_if}} expected-note@123{{candidate function}}
}

void f3(int m) __attribute__((overloadable)) __attribute__((enable_if(m >= 0, "positive")));
void f3(int m) __attribute__((overloadable)) __attribute__((enable_if(m < 0, "negative")));
void test7() {
  void (*p1)(int) = &f3; // expected-error{{initializing 'void (*)(int)' with an expression of incompatible type '<overloaded function type>'}} expected-note@131{{candidate function made ineligible by enable_if}} expected-note@132{{candidate function made ineligible by enable_if}}
  void (*p2)(int) = f3; // expected-error{{initializing 'void (*)(int)' with an expression of incompatible type '<overloaded function type>'}} expected-note@131{{candidate function made ineligible by enable_if}} expected-note@132{{candidate function made ineligible by enable_if}}
  void *p3 = (void*)&f3; // expected-error{{address of overloaded function 'f3' does not match required type 'void'}} expected-note@131{{candidate function made ineligible by enable_if}} expected-note@132{{candidate function made ineligible by enable_if}}
  void *p4 = (void*)f3; // expected-error{{address of overloaded function 'f3' does not match required type 'void'}} expected-note@131{{candidate function made ineligible by enable_if}} expected-note@132{{candidate function made ineligible by enable_if}}
}
#endif
