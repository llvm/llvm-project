// RUN: %clang_cc1 %s -verify -fsyntax-only
typedef char T[4];

T foo(int n, int m) {  }  // expected-error {{cannot return array type}}

void foof(const char *, ...) __attribute__((__format__(__printf__, 1, 2))), barf (void);

int typedef validTypeDecl(void) { } // expected-error {{function definition declared 'typedef'}}

struct _zend_module_entry { }    // expected-error {{expected ';' after struct}}
int gv1;
typedef struct _zend_function_entry { } // expected-error {{expected ';' after struct}} \
                                        // expected-warning {{typedef requires a name}}
int gv2;

static void buggy(int *x) { }

// Type qualifiers.
typedef int f(void); 
typedef f* fptr;
const f* v1;         // expected-warning {{qualifier on function type 'f' (aka 'int (void)') has unspecified behavior}}
__restrict__ f* v2;  // expected-error {{restrict requires a pointer or reference ('f' (aka 'int (void)') is invalid)}}
__restrict__ fptr v3; // expected-error {{pointer to function type 'f' (aka 'int (void)') may not be 'restrict' qualified}}
f *__restrict__ v4;   // expected-error {{pointer to function type 'f' (aka 'int (void)') may not be 'restrict' qualified}}

restrict struct hallo; // expected-error {{restrict requires a pointer or reference}}

// PR6180
struct test1 {
} // expected-error {{expected ';' after struct}}

void test2(void) {}


// PR6423
struct test3s {
} // expected-error {{expected ';' after struct}}
typedef int test3g;

// PR8264
const const int pr8264_1 = 0;  // expected-warning {{duplicate 'const' declaration specifier}}
volatile volatile int pr8264_2;  // expected-warning {{duplicate 'volatile' declaration specifier}}
char * restrict restrict pr8264_3;  // expected-warning {{duplicate 'restrict' declaration specifier}}

extern extern int pr8264_4;  // expected-warning {{duplicate 'extern' declaration specifier}}
void pr8264_5(void) {
  register register int x;  // expected-warning {{duplicate 'register' declaration specifier}}
}

inline inline void pr8264_6(void) {}  // expected-warning {{duplicate 'inline' declaration specifier}}
_Noreturn _Noreturn void pr8264_7(void);  // expected-warning {{duplicate '_Noreturn' declaration specifier}}
