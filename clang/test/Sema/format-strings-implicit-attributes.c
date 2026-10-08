// RUN: %clang_cc1 -std=c11 -ast-dump -Wno-gcc-compat -verify=expected,implicit %s | FileCheck %s --check-prefix=NO-FORMAT-ARG
// RUN: %clang_cc1 -std=c11 -ast-dump -Wno-gcc-compat -DNO_PROTOTYPE -verify %s | FileCheck %s --check-prefix=NO-FORMAT-ARG
// RUN: %clang_cc1 -std=c11 -ast-dump -Wno-gcc-compat -DMISSING_FORMAT_PARAM -verify %s | FileCheck %s --check-prefix=NO-FORMAT-ARG
// RUN: %clang_cc1 -std=c11 -ast-dump -DCFSTRING_VALID_PROTO -verify %s | FileCheck %s --check-prefix=FORMAT-ARG
// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wno-gcc-compat -DCFSTRING_INT_PARAM -verify %s

#ifdef CFSTRING_VALID_PROTO
// expected-no-diagnostics

// Keep inferring format_arg(1) when the referenced parameter exists.
char *__CFStringMakeConstantString(const char *);

// FORMAT-ARG-LABEL: FunctionDecl{{.*}} __CFStringMakeConstantString
// FORMAT-ARG: FormatArgAttr{{.*}}Implicit 1
#elif defined(CFSTRING_INT_PARAM)

// A non-string parameter must not crash format checking.
char *__CFStringMakeConstantString(int);

void a(char *) __attribute__((format(__CFString__, 1, 2)));

void b(void) {
  a(__CFStringMakeConstantString(1)); // expected-warning {{format string is not a string literal (potentially insecure)}}
  // expected-note@-1 {{treat the string as an argument to avoid this}}
}
#else

// Do not infer format attributes without a prototype or the format parameter.
// https://github.com/llvm/llvm-project/issues/225034
#if defined(MISSING_FORMAT_PARAM)
char *__CFStringMakeConstantString(void);
int asprintf(char **);
int vasprintf(char **);

void test_missing_format_param(char **out) {
  asprintf(out);
  vasprintf(out);
}
#elif defined(NO_PROTOTYPE)
char *__CFStringMakeConstantString();
int asprintf();
int vasprintf();

void test_no_prototype(void) {
  asprintf();
  vasprintf();
}
#endif

void a(char *) __attribute__((format(__CFString__, 1, 2))); // implicit-note {{passing argument to parameter here}}

void b(void) {
  a(__CFStringMakeConstantString()); // expected-warning {{format string is not a string literal (potentially insecure)}}
  // expected-note@-1 {{treat the string as an argument to avoid this}}
  // implicit-error@-2 {{call to undeclared function '__CFStringMakeConstantString'}}
  // implicit-error@-3 {{incompatible integer to pointer conversion passing 'int' to parameter of type 'char *'}}
}

// NO-FORMAT-ARG-NOT: FormatArgAttr
// NO-FORMAT-ARG: FunctionDecl{{.*}} b 'void (void)'
// NO-FORMAT-ARG-NOT: FormatArgAttr
#endif
