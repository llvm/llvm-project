
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
// Test incomplete unions

extern int external_union_len;
typedef union incomplete_union incomplete_union_t; // expected-note 2{{consider providing a complete definition for 'union incomplete_union'}}
extern incomplete_union_t* __counted_by(external_union_len) incompleteUnionPtr; // OK
extern union incomplete_union* __counted_by(external_union_len) incompleteUnionPtr2; // OK

int global_union_len;
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBUnionPtrImplicitInit' with type 'union incomplete_union *__single __counted_by(global_union_len)' (aka 'union incomplete_union *__single') because the pointee type 'union incomplete_union' is incomplete; consider providing a complete definition for 'union incomplete_union' before this definition or using the '__sized_by' attribute}}
union incomplete_union* __counted_by(global_union_len) GlobalCBUnionPtrImplicitInit; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONUnionPtrImplicitInit' with type 'union incomplete_union *__single __counted_by_or_null(global_union_len)' (aka 'union incomplete_union *__single') because the pointee type 'union incomplete_union' is incomplete; consider providing a complete definition for 'union incomplete_union' before this definition or using the '__sized_by_or_null' attribute}}
union incomplete_union* __counted_by_or_null(global_union_len) GlobalCBONUnionPtrImplicitInit; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

// Unions are handled like structs for the diagnostics so the testing for structs
// should mean testing other combinations should be unnecessary.
