
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

void Test(int sel) {
    char c;
    char *x = &c;
    char *y = &c;
    char *z = sel ? x : y;

    char * __null_terminated x_nt = __unsafe_forge_null_terminated(char *, x);
    char * __null_terminated y_nt = __unsafe_forge_null_terminated(char *, y);
    z = sel ? x_nt : y_nt; // expected-error{{assigning to 'char *__bidi_indexable' from incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
                           // expected-note@-5{{consider adding '__null_terminated' to 'z'}}
                           // expected-note@-2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
                           // expected-note@-3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
    z = sel ? x : y_nt;    // expected-error{{converting 'char *__bidi_indexable' to incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
                           // expected-note@-1{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
                           // expected-note@-2{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}

    char * __null_terminated z_nt = sel ? x_nt : y_nt;
    z_nt = sel ? x_nt : y; // expected-error{{converting 'char *__bidi_indexable' to incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
                           // expected-note@-1{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
                           // expected-note@-2{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    z_nt = sel ? x : y_nt; // expected-error{{converting 'char *__bidi_indexable' to incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
                           // expected-note@-1{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
                           // expected-note@-2{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}

    char * __single y_single = &c;
    char * __single x_single = &c;
    z_nt = sel ? x_nt : y_single; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single __terminated_by(0)' (aka 'char *__single') and 'char *__single'}}
    z_nt = sel ? x_single : y_nt; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single' and 'char *__single __terminated_by(0)' (aka 'char *__single')}}

    char * __terminated_by(2) x_2t = __unsafe_forge_terminated_by(char *, x, 2);
    z_nt = sel ? x_2t : y_nt; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single __terminated_by(2)' (aka 'char *__single') and 'char *__single __terminated_by(0)' (aka 'char *__single')}}
    z_nt = sel ? x_nt : x_2t; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single __terminated_by(0)' (aka 'char *__single') and 'char *__single __terminated_by(2)' (aka 'char *__single')}}

    // Type comparison takes a different path when the pointee types are not the same, exercise this
    const char * __null_terminated x_c_nt = x_nt;
    z_nt = sel ? x_c_nt : y_nt; // expected-warning{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from 'const char *__single __terminated_by(0)' (aka 'const char *__single') discards qualifiers}}
    z_nt = sel ? x_nt : x_c_nt; // expected-warning{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from 'const char *__single __terminated_by(0)' (aka 'const char *__single') discards qualifiers}}

    z_nt = sel ? x_c_nt : x_2t; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'const char *__single __terminated_by(0)' (aka 'const char *__single') and 'char *__single __terminated_by(2)' (aka 'char *__single')}}
    z_nt = sel ? x_2t : x_c_nt; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single __terminated_by(2)' (aka 'char *__single') and 'const char *__single __terminated_by(0)' (aka 'const char *__single')}}

    z_nt = sel ? x_c_nt : x_single; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'const char *__single __terminated_by(0)' (aka 'const char *__single') and 'char *__single'}}
    z_nt = sel ? x_single : x_c_nt; // expected-error{{conditional expression evaluates values with mismatching __terminated_by attributes 'char *__single' and 'const char *__single __terminated_by(0)' (aka 'const char *__single')}}

    unsigned char * __null_terminated x_u_nt = x_nt; // expected-warning{{initializing 'unsigned char *__single __terminated_by(0)' (aka 'unsigned char *__single') with an expression of type 'char *__single __terminated_by(0)' (aka 'char *__single') converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
    signed char * __null_terminated y_s_nt = y_nt; // expected-warning{{initializing 'signed char *__single __terminated_by(0)' (aka 'signed char *__single') with an expression of type 'char *__single __terminated_by(0)' (aka 'char *__single') converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
    z_nt = sel ? x_u_nt : y_s_nt; // expected-error{{conditional expression evaluates values with incompatible pointee types 'unsigned char *__single __terminated_by(0)' (aka 'unsigned char *__single') and 'signed char *__single __terminated_by(0)' (aka 'signed char *__single'); use explicit casts to perform this conversion}}
    z_nt = sel ? y_s_nt : x_u_nt; // expected-error{{conditional expression evaluates values with incompatible pointee types 'signed char *__single __terminated_by(0)' (aka 'signed char *__single') and 'unsigned char *__single __terminated_by(0)' (aka 'unsigned char *__single'); use explicit casts to perform this conversion}}
    z_nt = sel ? y_s_nt : (signed char * __null_terminated)x_u_nt; // expected-warning{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from 'signed char *__single __terminated_by(0)' (aka 'signed char *__single') converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
    z_nt = sel ? (signed char * __single _Nullable __null_terminated) y_s_nt : (signed char * _Nullable __null_terminated)x_u_nt; // expected-warning{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from 'signed char *__single __terminated_by(0) _Nullable' (aka 'signed char *__single') converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
}
