
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -I%S/SystemHeaders/include -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -I%S/SystemHeaders/include -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <static-bound-ptr-init.h>
struct T {
    void (*fp)(const struct T *t);
    int i;
};

static void foo(const struct T *t) {}

const struct T t = {
    .fp = foo,
    .i = 0,
};

// BoundsSafetyPointerCast Evaluation Results
// Types - B: __bidi_indexable, T: __single, U: __unsafe_indexable, A: __indexable
//         S: CString, C: Count
// Result - P: Compile-time constant, F: Compile-time error,
//          R: No compile-time constant, Z: P and Zero-bound,
//          D: Depending on the metadata
// * (at least one elem)
// ^ (upper bounds check)
// < (if the destination type size is smaller)

// To  | B | T | U | A | S | C |
// From|   |   |   |   |   |   |
// ----------------------------
// B   | P | R | P | R | R | R |
// T   | P | P<| P | P | Z | RD|
// U   | F | P | P | F | F | F |
// A   | P | R | P | P | R | R |
// S   | P | P | P | P*| P | R |
// C   | P | D | P | P | R | R |

int g_var = 0;

// B -> B: P
int *__bidi_indexable bi_p = &g_var;
// B -> A: R^
int *__indexable fi_p = &g_var;
// B -> T: R
int *__single s_p = &g_var;
// B -> U: P
int *__unsafe_indexable ui_p = &g_var;

// T -> B: P
// expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type}}
void *__bidi_indexable bi_p3 = foo; // expected-note{{pointer 'bi_p3' declared here}}
// T -> A: P
// expected-error-re@+1{{cannot initialize indexable pointer {{.+}} from __single pointer to incomplete type}}
void *__indexable fi_p3 = foo; // expected-note{{pointer 'fi_p3' declared here}}
// T -> T: P
void *__single s_p3 = foo;
// T -> U: P
void *__unsafe_indexable ui_p2 = foo;

void Test () {
    // Special case: "unspecified" should convert to __unsafe_indexable
    const char *__unsafe_indexable ui_p3 = mock_system_func();

    static int len1;
    // B -> C: R
    // expected-warning@+1{{possibly initializing 'c_p1' of type 'int *__single __counted_by(len1)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
    static int *__counted_by(len1) c_p1 = &g_var; // expected-error{{initializer element is not a compile-time constant}}
    static int len2;
    // T -> C: RD

#pragma mark - UNEXPECTEDLY PASSES
    // expected-warning@+1{{possibly initializing 'c_p2' of type 'void *__single __sized_by(len2)' (aka 'void *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
    static void *__sized_by(len2) c_p2 = foo; // expected-error{{initializer element is not a compile-time constant}}
    static int len3 = 1;
    // T -> C: RD
    // expected-error@+1{{initializing 'c_p3' of type 'void *__single __sized_by(len3)' (aka 'void *__single') and size value of 1 with 'void (*__single)(const struct T *__single)' and pointee of size 0 always fails}}
    static void *__sized_by(len3) c_p3 = foo; // expected-error{{initializer element is not a compile-time constant}}
#pragma mark -

    static int s_var = 0;
    static int *__counted_by(10) c_p4 = &s_var; // expected-error{{initializer element is not a compile-time constant}}

    static int s_arr[10];
    static int *__counted_by(10) c_p5 = s_arr; // ok

    static const int len4 = 1;
    static int *__counted_by(len4) c_p6 = &g_var; // ok
}

struct RangedStruct {
  int *__ended_by(end) start;
  int *end;
};

// TODO
void TestEndedBy() {
  // expected-error@+2{{initializer element is not a compile-time constant}}
  static struct RangedStruct rs = {
    0, 0
  };

  static char s_arr[10];
  // expected-error@+1{{initializer element is not a compile-time constant}}
  static char *end = &s_arr[10];
  // expected-error@+1{{initializer element is not a compile-time constant}}
  static char *__ended_by(end) start = &s_arr[0];
}
