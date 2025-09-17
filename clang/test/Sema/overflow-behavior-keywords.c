// RUN: %clang_cc1 -fsyntax-only -foverflow-behavior-types -verify %s

int __wrap a;
int __no_wrap b;

const int __wrap c;
volatile int __no_wrap d;
const volatile int __wrap e;

int __attribute__((overflow_behavior(wrap))) attr_style_var;
int __wrap keyword_style_var;

int __wrap __no_wrap conflicting_var; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}

// Test duplicate qualifiers
int __wrap __wrap duplicate_wrap; // expected-warning{{duplicate '__wrap' declaration specifier}}
int __no_wrap __no_wrap duplicate_no_wrap; // expected-warning{{duplicate '__no_wrap' declaration specifier}}

int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(no_wrap))) attr_conflict; // expected-warning{{conflicting 'overflow_behavior' attributes on the same type; 'no_wrap' takes precedence over 'wrap'}}

// Test duplicate attributes - no warning, less problematic than duplicate decl specifiers
int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(wrap))) duplicate_attr;

const volatile int __wrap __no_wrap __wrap complex_conflict; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}} expected-warning{{duplicate '__wrap' declaration specifier}}

extern int __wrap __no_wrap extern_conflict; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}
static int __wrap __no_wrap static_conflict; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}

void test_storage_class(void) {
    register int __wrap __no_wrap register_conflict; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}
}

int __wrap __no_wrap *ptr_conflict; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}
int __wrap __no_wrap arr_conflict[5]; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}

int __wrap __no_wrap (*func_ptr_conflict)(void); // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}

void param_test(int __wrap __no_wrap param); // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}

struct conflict_struct {
    int __wrap __no_wrap member; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}
    int __wrap __wrap dup_member; // expected-warning{{duplicate '__wrap' declaration specifier}}
};

typedef int __wrap __no_wrap conflict_typedef; // expected-warning{{conflicting overflow behavior specifiers on the same type; 'no_wrap' takes precedence over 'wrap'}}
typedef int __wrap __wrap dup_typedef; // expected-warning{{duplicate '__wrap' declaration specifier}}

int __wrap *ptr_to_wrap;
int __no_wrap arr[10];
int __wrap (*func_ptr)(int);

void test_function(int __wrap param1, int __no_wrap param2);

struct test_struct {
    int __wrap member1;
    int __no_wrap member2;
};

typedef int __wrap wrap_int_t;
typedef int __no_wrap no_wrap_int_t;

typedef float __wrap float_wrap; // expected-warning{{__wrap specifier cannot be applied to non-integer type 'float'; specifier ignored}}
typedef double __no_wrap double_no_wrap; // expected-warning{{__no_wrap specifier cannot be applied to non-integer type 'double'; specifier ignored}}

struct S { int i; };
typedef struct S __wrap struct_wrap; // expected-warning{{__wrap specifier cannot be applied to non-integer type 'struct S'; specifier ignored}}
