
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
// Test incomplete enums

extern int external_enum_len;
typedef enum incomplete_enum incomplete_enum_t; // expected-note 2{{consider providing a complete definition for 'enum incomplete_enum'}}
extern incomplete_enum_t* __counted_by(external_enum_len) incompleteEnumPtr; // OK
extern enum incomplete_enum* __counted_by(external_enum_len) incompleteEnumPtr2; // OK

int global_enum_len;
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBEnumPtrImplicitInit' with type 'enum incomplete_enum *__single __counted_by(global_enum_len)' (aka 'enum incomplete_enum *__single') because the pointee type 'enum incomplete_enum' is incomplete; consider providing a complete definition for 'enum incomplete_enum' before this definition or using the '__sized_by' attribute}}
enum incomplete_enum* __counted_by(global_enum_len) GlobalCBEnumPtrImplicitInit; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONEnumPtrImplicitInit' with type 'enum incomplete_enum *__single __counted_by_or_null(global_enum_len)' (aka 'enum incomplete_enum *__single') because the pointee type 'enum incomplete_enum' is incomplete; consider providing a complete definition for 'enum incomplete_enum' before this definition or using the '__sized_by_or_null' attribute}}
enum incomplete_enum* __counted_by_or_null(global_enum_len) GlobalCBONEnumPtrImplicitInit; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}


// Unions are handled like structs for the diagnostics so the testing for structs
// should mean testing other combinations should be unnecessary.
