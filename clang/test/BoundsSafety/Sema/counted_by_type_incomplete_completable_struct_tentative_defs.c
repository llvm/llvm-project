
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
// Test diagnostics on _counted_by(_or_null) pointers with an incomplete struct
// pointee type **and** that involve tentative definitions.

// NOTE: For a typedef the source location is of the underlying type instead of
// the typedef. This seems like the right behavior because the typedef isn't the
// forward declaration, `struct IncompleteStructTy` is.
//
// expected-note@+2 2{{consider providing a complete definition for 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy')}}
// expected-note@+1 30{{consider providing a complete definition for 'struct IncompleteStructTy'}}
struct IncompleteStructTy;

//------------------------------------------------------------------------------
// Only one tentative definition
//------------------------------------------------------------------------------

// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_one' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
struct IncompleteStructTy*__counted_by(0) cb_t_dfn_one; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_one' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_one; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_cb_t_dfn_one(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_one' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_one = 0x0;
}

void use_cbon_t_dfn_one(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_one' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_one = 0x0;
}

//------------------------------------------------------------------------------
// Two tentative definitions
//------------------------------------------------------------------------------

// We only error on the last one which is treated as a definition once the whole
// TU has been processed.

struct IncompleteStructTy*__counted_by(0) cb_t_dfn_two; // OK - tentative definition acts like declaration
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_two' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
struct IncompleteStructTy*__counted_by(0) cb_t_dfn_two; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}

struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_two; // OK - tentative definition acts like declaration
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_two' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_two; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_cb_t_dfn_two(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_two' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_two = 0x0;
}

void use_cbon_t_dfn_two(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_two' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_two = 0x0;
}

//------------------------------------------------------------------------------
// Definition followed by tentative definition
//------------------------------------------------------------------------------

// NOTE: The diagnostic about initializing `cb_t_dfn_after_def` is suppressed.

// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'cb_t_dfn_after_def' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
struct IncompleteStructTy*__counted_by(0) cb_t_dfn_after_def = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
struct IncompleteStructTy*__counted_by(0) cb_t_dfn_after_def; // OK - tentative definition acts like declaration
// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
extern struct IncompleteStructTy*__counted_by(0) cb_t_dfn_after_def; // OK - declaration

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'cbon_t_dfn_after_def' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_after_def = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_after_def; // OK - tentative definition acts like declaration
// expected-note@+1{{__counted_by_or_null}}
extern struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_after_def; // OK - declaration

void use_cb_t_dfn_after_def(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_after_def' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_after_def = 0x0;
}

void use_cbon_t_dfn_after_def(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_after_def' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_after_def = 0x0;
}

//------------------------------------------------------------------------------
// Definition preceeded by tentative definition
//------------------------------------------------------------------------------

// NOTE: The diagnostic about initializing `cb_t_dfn_after_def` is suppressed.
// NOTE: This test case is the **only** case where diagnostics about variable use
// are suppressed due the variable decl being invalid. So it does more testing
// than other `use_*` functions in this file.

struct IncompleteStructTy*__counted_by(0) cb_t_dfn_before_def; // OK - tentative definition acts like declaration
// expected-note@+1 2{{consider using '__sized_by' instead of '__counted_by'}}
extern struct IncompleteStructTy*__counted_by(0) cb_t_dfn_before_def; // OK - declaration

void consume_cb_const_count_zero(struct IncompleteStructTy*__counted_by(0) p);

void use_cb_t_dfn_before_def_before_def(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_before_def' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_before_def = 0x0;

  // expected-error@+1{{cannot use 'cb_t_dfn_before_def' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_cb_const_count_zero(cb_t_dfn_before_def);
}

// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'cb_t_dfn_before_def' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
struct IncompleteStructTy*__counted_by(0) cb_t_dfn_before_def = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

void use_cb_t_dfn_before_def_after_def(void) {
  // no error here. At this point `cb_t_dfn_before_def` has been marked as invalid
  // because we've seen a definition and have marked the variable as invalid.
  cb_t_dfn_before_def = 0x0;
  consume_cb_const_count_zero(cb_t_dfn_before_def);
}

struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_before_def; // OK - tentative definition acts like declaration
// expected-note@+1 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
extern struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_before_def; // OK - declaration

void consume_cbon_const_count_zero(struct IncompleteStructTy*__counted_by_or_null(0) p);

void use_cbon_t_dfn_before_def_before_def(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_before_def' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_before_def = 0x0;

  // expected-error@+1{{cannot use 'cbon_t_dfn_before_def' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_cbon_const_count_zero(cbon_t_dfn_before_def);
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'cbon_t_dfn_before_def' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_before_def = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_cbon_t_dfn_before_def_after_def(void) {
  // no error here. At this point `cbon_t_dfn_before_def` has been marked as invalid
  // because we've seen a definition and have marked the variable as invalid.
  cbon_t_dfn_before_def = 0x0;

  consume_cbon_const_count_zero(cbon_t_dfn_before_def);
}

//------------------------------------------------------------------------------
// Tentative definition due to `static` keyword
//------------------------------------------------------------------------------

static struct IncompleteStructTy*__counted_by(0) cb_t_dfn_static_def; // OK - tentative definition acts like declaration
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_static_def' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
static struct IncompleteStructTy*__counted_by(0) cb_t_dfn_static_def; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}

static struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_static_def; // OK - tentative definition acts like declaration
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_static_def' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
static struct IncompleteStructTy*__counted_by_or_null(0) cbon_t_dfn_static_def; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_cb_t_dfn_static_def(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_static_def' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_static_def = 0x0;
}

void use_cbon_t_dfn_static_def(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_static_def' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_static_def = 0x0;
}

//------------------------------------------------------------------------------
// Tentative definition and non-const count
//------------------------------------------------------------------------------

static int cb_t_dfn_static_non_const_count_count;
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_static_non_const_count' with type 'struct IncompleteStructTy *__single __counted_by(cb_t_dfn_static_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
static struct IncompleteStructTy*__counted_by(cb_t_dfn_static_non_const_count_count) cb_t_dfn_static_non_const_count; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}

static int cbon_t_dfn_static_non_const_count_count;
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_static_non_const_count' with type 'struct IncompleteStructTy *__single __counted_by_or_null(cbon_t_dfn_static_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
static struct IncompleteStructTy*__counted_by_or_null(cbon_t_dfn_static_non_const_count_count) cbon_t_dfn_static_non_const_count; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}


int cb_t_dfn_non_const_count_count;
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_non_const_count' with type 'struct IncompleteStructTy *__single __counted_by(cb_t_dfn_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by' attribute}}
struct IncompleteStructTy*__counted_by(cb_t_dfn_non_const_count_count) cb_t_dfn_non_const_count; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}

int cbon_t_dfn_non_const_count_count;
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_non_const_count' with type 'struct IncompleteStructTy *__single __counted_by_or_null(cbon_t_dfn_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete; consider providing a complete definition for 'struct IncompleteStructTy' before this definition or using the '__sized_by_or_null' attribute}}
struct IncompleteStructTy*__counted_by_or_null(cbon_t_dfn_non_const_count_count) cbon_t_dfn_non_const_count; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_cb_t_dfn_static_non_const_count(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_static_non_const_count' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(cb_t_dfn_static_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_static_non_const_count = 0x0;
  cb_t_dfn_static_non_const_count_count = 0;
}

void use_cbon_t_dfn_static_non_const_count(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_static_non_const_count' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(cbon_t_dfn_static_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_static_non_const_count = 0x0;
  cbon_t_dfn_static_non_const_count_count = 0;
}

void use_cb_t_dfn_non_const_count(void) {
  // expected-error@+1{{cannot assign to 'cb_t_dfn_non_const_count' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(cb_t_dfn_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_t_dfn_non_const_count = 0x0;
  cb_t_dfn_non_const_count_count = 0;
}

void use_cbon_t_dfn_non_const_count(void) {
  // expected-error@+1{{cannot assign to 'cbon_t_dfn_non_const_count' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(cbon_t_dfn_non_const_count_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_t_dfn_non_const_count = 0x0;
  cbon_t_dfn_non_const_count_count = 0;
}

//------------------------------------------------------------------------------
// Only one tentative definition on typedef
//------------------------------------------------------------------------------

typedef struct IncompleteStructTy Incomplete_Struct_t;

// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'cb_t_dfn_one_typedef' with type 'Incomplete_Struct_t *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete; consider providing a complete definition for 'Incomplete_Struct_t' before this definition or using the '__sized_by' attribute}}
Incomplete_Struct_t*__counted_by(0) cb_t_dfn_one_typedef; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'cbon_t_dfn_one_typedef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete; consider providing a complete definition for 'Incomplete_Struct_t' before this definition or using the '__sized_by_or_null' attribute}}
Incomplete_Struct_t*__counted_by_or_null(0) cbon_t_dfn_one_typedef; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}


//------------------------------------------------------------------------------
// Tentative definition using incomplete pointee that is later completed
//------------------------------------------------------------------------------

struct IncompleteStructLaterCompletedTy;

// This is a tentative definition where the pointee type is incomplete. However,
// by the end of the translation unit (when we do the check) the
// pointee type has now become complete so the error diagnostic is not emitted.
struct IncompleteStructLaterCompletedTy* __counted_by(0) cb_t_dfn_later_completed;

struct IncompleteStructLaterCompletedTy {
  int field;
};

//------------------------------------------------------------------------------
// Tentative definition of struct with field using __counted_by or
// __counted_by_or_null with an incomplete pointee type
//------------------------------------------------------------------------------
struct BuffersCBTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by(count) buffer;
  Incomplete_Struct_t* __counted_by(count_typedef) buffer_typedef;
};

struct BuffersCBONTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by_or_null(count) buffer;
  Incomplete_Struct_t* __counted_by_or_null(count_typedef) buffer_typedef;
};

// TODO: Technically this is ok because zero initialization doesn't require
// a check and therefor doesn't need the pointee type size. However, it's
// inconsistent to disallow assignment via `= {0}` but allow it for tentative
// definitions that get zero initialized. rdar://133573722
struct BuffersCBTy GlobalBuffersCBTy_implicit_init;
struct BuffersCBONTy GlobalBuffersCBONTy_implicit_init;

struct BuffersNonZeroCountCBTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by(count+1) buffer;
  Incomplete_Struct_t* __counted_by(count_typedef+1) buffer_typedef;
};

struct BuffersNonZeroCountCBONTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by_or_null(count+1) buffer;
  Incomplete_Struct_t* __counted_by_or_null(count_typedef+1) buffer_typedef;
};

// TODO: It's inconsistent to disallow assignment via `= {0}` but allow it for
// tentative definitions that get zero initialized. rdar://133573722
//
// expected-error@+2{{implicitly initializing 'GlobalBuffersNonZeroCountCBTy_implicit_init.buffer' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
// expected-error@+1{{implicitly initializing 'GlobalBuffersNonZeroCountCBTy_implicit_init.buffer_typedef' of type 'Incomplete_Struct_t *__single __counted_by(count_typedef + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
struct BuffersNonZeroCountCBTy GlobalBuffersNonZeroCountCBTy_implicit_init;
struct BuffersNonZeroCountCBONTy GlobalBuffersNonZeroCountCBONTy_implicit_init; // OK
