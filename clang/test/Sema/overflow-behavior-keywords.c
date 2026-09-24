// RUN: %clang_cc1 -fsyntax-only -fexperimental-overflow-behavior-types -verify %s

int __ob_wrap a;
int __ob_trap b;

const int __ob_wrap c;
volatile int __ob_trap d;
const volatile int __ob_wrap e;

int __attribute__((overflow_behavior(wrap))) attr_style_var;
int __ob_wrap keyword_style_var;

int __ob_wrap __ob_trap conflicting_var; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}

// Test duplicate qualifiers
int __ob_wrap __ob_wrap duplicate_wrap; // expected-warning{{duplicate '__ob_wrap' declaration specifier}}
int __ob_trap __ob_trap duplicate_trap; // expected-warning{{duplicate '__ob_trap' declaration specifier}}

int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(trap))) attr_conflict; // expected-error{{conflicting 'overflow_behavior' attributes on the same type}}

// Test duplicate attributes - no warning, less problematic than duplicate decl specifiers
int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(wrap))) duplicate_attr;

const volatile int __ob_wrap __ob_trap __ob_wrap complex_conflict; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}} expected-warning{{duplicate '__ob_wrap' declaration specifier}}

extern int __ob_wrap __ob_trap extern_conflict; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}
static int __ob_wrap __ob_trap static_conflict; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}

void test_storage_class(void) {
    register int __ob_wrap __ob_trap register_conflict; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}
}

int __ob_wrap __ob_trap *ptr_conflict; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}
int __ob_wrap __ob_trap arr_conflict[5]; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}

int __ob_wrap __ob_trap (*func_ptr_conflict)(void); // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}

void param_test(int __ob_wrap __ob_trap param); // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}

struct conflict_struct {
    int __ob_wrap __ob_trap member; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}
    int __ob_wrap __ob_wrap dup_member; // expected-warning{{duplicate '__ob_wrap' declaration specifier}}
};

typedef int __ob_wrap __ob_trap conflict_typedef; // expected-error{{cannot combine with previous '__ob_wrap' declaration specifier}}
typedef int __ob_wrap __ob_wrap dup_typedef; // expected-warning{{duplicate '__ob_wrap' declaration specifier}}

int __ob_wrap *ptr_to_wrap;
int __ob_trap arr[10];
int __ob_wrap (*func_ptr)(int);

void test_function(int __ob_wrap param1, int __ob_trap param2);

struct test_struct {
    int __ob_wrap member1;
    int __ob_trap member2;
};

typedef int __ob_wrap wrap_int_t;
typedef int __ob_trap trap_int_t;

typedef float __ob_wrap float_wrap; // expected-error{{__ob_wrap specifier cannot be applied to non-integer type 'float'}}
typedef double __ob_trap double_trap; // expected-error{{__ob_trap specifier cannot be applied to non-integer type 'double'}}

struct S { int i; };
typedef struct S __ob_wrap struct_wrap; // expected-error{{__ob_wrap specifier cannot be applied to non-integer type 'struct S'}}

__ob_trap struct S2 { // expected-error{{__ob_trap specifier cannot be applied to non-integer type 'struct S2'}}
  int a;
} s2;
