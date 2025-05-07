// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,rs %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,rs %s
#include <ptrcheck.h>
// Test diagnostics on _counted_by(_or_null) pointers with an incomplete struct
// pointee type.

// NOTE: For a typedef the source location is of the underlying type instead of
// the typedef. This seems like the right behavior because the typedef isn't the
// forward declaration, `struct IncompleteStructTy` is.
//
// expected-note@+1 62{{consider providing a complete definition for 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy')}}
struct IncompleteStructTy; // expected-note 153{{consider providing a complete definition for 'struct IncompleteStructTy'}}

typedef struct IncompleteStructTy Incomplete_Struct_t;

//------------------------------------------------------------------------------
// Attribute on parameters
//------------------------------------------------------------------------------

// On declarations its ok to use the attribute
void no_consume_ok(
  struct IncompleteStructTy* __counted_by(size) cb,
  struct IncompleteStructTy* __counted_by_or_null(size) cbon,
  int size); // OK

// Using the attribute on parameters on a function **definition** is not allowed.
void no_consume_ok(
  // expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'cb' with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(size) cb, // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'cbon' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(size) cbon, // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int size) {

}

void no_consume_ok_unnamed_param(
  struct IncompleteStructTy* __counted_by(size),
  struct IncompleteStructTy* __counted_by_or_null(size),
  int size); // OK

void no_consume_ok_unnamed_param(
  // expected-error@+2{{cannot apply '__counted_by' attribute to parameter with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-warning@+1{{omitting the parameter name in a function definition is a C23 extension}}
  struct IncompleteStructTy* __counted_by(size), // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  // expected-error@+2{{cannot apply '__counted_by_or_null' attribute to parameter with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-warning@+1{{omitting the parameter name in a function definition is a C23 extension}}
  struct IncompleteStructTy* __counted_by_or_null(size), // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int size) {

}

void consume_cb(struct IncompleteStructTy* __counted_by(size_cb), int size_cb);
void consume_cbon(struct IncompleteStructTy* __counted_by_or_null(size_cbon), int size_cbon);

void consume_param_read_write(
  // expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'cb' with type 'struct IncompleteStructTy *__single __counted_by(size_cb)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(size_cb) cb, // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'cbon' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size_cbon)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(size_cbon) cbon, // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int size_cb,
  int size_cbon) {

  // There shouldn't be diagnostics on the uses because the parameters are marked
  // as invalid.

  // Read
  struct IncompleteStructTy* local = cb;
  local = cb;
  local =&cb[1];
  consume_cb(cb);

  // Write
  cb = 0x0;
  // TODO: This diagnostic should not be firing. rdar://133001202
  // expected-error@+1{{assignment to 'size_cb' requires corresponding assignment to 'struct IncompleteStructTy *__single __counted_by(size_cb)' (aka 'struct IncompleteStructTy *__single') 'cb'; add self assignment 'cb = cb' if the value has not changed}}
  size_cb = 0;


  // Read
  struct IncompleteStructTy* local2 = cbon;
  local2 = cbon;
  local2 =&cbon[1];
  consume_cbon(cbon);

  // Write
  cbon = 0x0;
  // TODO: This diagnostic should not be firing. rdar://133001202
  // expected-error@+1{{assignment to 'size_cbon' requires corresponding assignment to 'struct IncompleteStructTy *__single __counted_by_or_null(size_cbon)' (aka 'struct IncompleteStructTy *__single') 'cbon'; add self assignment 'cbon = cbon' if the value has not changed}}
  size_cbon = 0;
}

// These errors seem prevent emitting any further diagnostics about the attributes.
void no_consume_default_assign(
  // expected-error@+1{{C does not support default arguments}}
  struct IncompleteStructTy* __counted_by(size) cb = 0x0,
  // expected-error@+1{{C does not support default arguments}}
  struct IncompleteStructTy* __counted_by_or_null(size) cbon = 0x0,
  int size) {

}

//------------------------------------------------------------------------------
// Attribute on parameters with nested attributes
//------------------------------------------------------------------------------
void consume_param_nested(
  struct IncompleteStructTy* __counted_by(size1)* cb, // expected-note 3{{consider using '__sized_by' instead of '__counted_by'}}
  struct IncompleteStructTy* __counted_by_or_null(size2)* cbon, // expected-note 3{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int size1, int size2) {

  // Surprisingly `&cb[0]` doesn't count as a use.
  struct IncompleteStructTy** local = &cb[0];

  // expected-error@+1{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  local = cb;

  // expected-error@+1{{cannot assign to object with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  *cb = 0x0;
  size1 = 0;

  // expected-error@+1{{cannot use 'cb[0]' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  void* read_cb = cb[0];
  // expected-error@+1{{cannot use '*cb' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_cb(*cb, size1);

  // expected-error@+1{{not allowed to change out parameter with dependent count}}
  cb = 0x0;

  // Surprisingly `&cbon[0]` doesn't count as a use.
  struct IncompleteStructTy** local2 = &cbon[0];
  // expected-error@+1{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  local = cbon;

  // expected-error@+1{{cannot assign to object with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size2)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  *cbon = 0x0;
  size2 = 0;

  // expected-error@+1{{cannot use 'cbon[0]' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size2)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  void* read_cbon = cbon[0];
  // expected-error@+1{{cannot use '*cbon' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size2)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_cbon(*cbon, size2);

  // expected-error@+1{{not allowed to change out parameter with dependent count}}
  cbon = 0x0;
}

//------------------------------------------------------------------------------
// Attribute on return type of called function
//------------------------------------------------------------------------------

// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
struct IncompleteStructTy* __counted_by(size) ret_cb_IncompleteStructTy(int size); // OK
// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
Incomplete_Struct_t* __counted_by(size) ret_cb_IncompleteStructTy_typedef(int size); // OK
// expected-note@+1{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
struct IncompleteStructTy* __counted_by_or_null(size) ret_cbon_IncompleteStructTy(int size); // OK
// expected-note@+1{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
Incomplete_Struct_t* __counted_by_or_null(size) ret_cbon_IncompleteStructTy_typedef(int size); // OK

// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
struct IncompleteStructTy* __counted_by(1) ret_cb_IncompleteStructTy_const_count_one(void); // OK
// expected-note@+1{{consider using '__sized_by' instead of '__counted_by'}}
Incomplete_Struct_t* __counted_by(1) ret_cb_IncompleteStructTy_typedef_const_count_one(void); // OK
// expected-note@+1{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
struct IncompleteStructTy* __counted_by_or_null(1) ret_cbon_IncompleteStructTy_const_count_one(void); // OK
// expected-note@+1{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
Incomplete_Struct_t* __counted_by_or_null(1) ret_cbon_IncompleteStructTy_typedef_const_count_one(void); // OK

void call_fn_returns_incomplete_pointee(void) {
  int size = 0;
  // expected-error@+1{{cannot call 'ret_cb_IncompleteStructTy' with '__counted_by' attributed return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  ret_cb_IncompleteStructTy(size);
  // expected-error@+1{{cannot call 'ret_cb_IncompleteStructTy_typedef' with '__counted_by' attributed return type 'Incomplete_Struct_t *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  ret_cb_IncompleteStructTy_typedef(size);
  // expected-error@+1{{cannot call 'ret_cbon_IncompleteStructTy' with '__counted_by_or_null' attributed return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  ret_cbon_IncompleteStructTy(size);
  // expected-error@+1{{cannot call 'ret_cbon_IncompleteStructTy_typedef' with '__counted_by_or_null' attributed return type 'Incomplete_Struct_t *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  ret_cbon_IncompleteStructTy_typedef(size);

  // expected-error@+1{{cannot call 'ret_cb_IncompleteStructTy_const_count_one' with '__counted_by' attributed return type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  ret_cb_IncompleteStructTy_const_count_one();
  // expected-error@+1{{cannot call 'ret_cb_IncompleteStructTy_typedef_const_count_one' with '__counted_by' attributed return type 'Incomplete_Struct_t *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  ret_cb_IncompleteStructTy_typedef_const_count_one();
  // expected-error@+1{{cannot call 'ret_cbon_IncompleteStructTy_const_count_one' with '__counted_by_or_null' attributed return type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  ret_cbon_IncompleteStructTy_const_count_one();
  // expected-error@+1{{cannot call 'ret_cbon_IncompleteStructTy_typedef_const_count_one' with '__counted_by_or_null' attributed return type 'Incomplete_Struct_t *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  ret_cbon_IncompleteStructTy_typedef_const_count_one();
}

//------------------------------------------------------------------------------
// Attribute on return type in function declaration
//------------------------------------------------------------------------------

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb(int size) {
  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_missing_return(int size) {
  // missing return statement
}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  consume_param_and_return_cbon_missing_return(int size) {
  // missing return statement
}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note 3{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_multiple_returns(int size) {
  if (size == 0)
    // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    return 0x0;

  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note 3{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  consume_param_and_return_cbon_multiple_returns(int size) {
  if (size == 0)
    // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    return 0x0;

  // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(1) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_const_count_1(int size) {
  // rs-error@+2{{returning null from a function with result type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 always fails}}
  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
consume_param_and_return_cb_single_forge(int size) {
  // rs-warning@+2{{count value is not statically known: returning 'struct IncompleteStructTy *__single' from a function with result type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') is invalid for any count other than 0 or 1}}
  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return __unsafe_forge_single(struct IncompleteStructTy*, 0x0);
}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_single_forge_bidi(int size) {
  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x0, 4);
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  consume_param_and_return_cbon(int size) {
  // TODO: We should consider allowing this because the assignment of nullptr
  // means the type size isn't needed (rdar://129424354).
  // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
consume_param_and_return_cbon_single_forge(int size) {
  // rs-warning@+2{{count value is not statically known: returning 'struct IncompleteStructTy *__single' from a function with result type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') is invalid for any count other than 0 or 1 unless the pointer is null}}
  // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return __unsafe_forge_single(struct IncompleteStructTy*, 0x0);
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  consume_param_and_return_cbon_single_forge_bidi(int size) {
  // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x0, 4);
}

// Test typedef as the incomplete pointee type
// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'Incomplete_Struct_t *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by(size) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_typedef(int size) {
  // expected-error@+1{{cannot return '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  return 0x0;
}

// Check Incomplete type diagnostic and bad conversion diagnostics both emitted on return

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(size) // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  consume_param_and_return_cb_bad_conversion(int size) {
  // expected-error@+2{{cannot return '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  return 0x1;
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(size) // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  consume_param_and_return_cbon_bad_conversion(int size) {
  // expected-error@+2{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{non-pointer to safe pointer conversion is not allowed with -fbounds-safety; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable'}}
  return 0x1;
}

//------------------------------------------------------------------------------
// Pass Arguments to parameters with attribute.
//------------------------------------------------------------------------------
void consume_incomplete_cb(struct IncompleteStructTy* __counted_by(size) c, int size); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
void consume_incomplete_cb_unnamed(struct IncompleteStructTy* __counted_by(size), int size); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
typedef void consume_incomplete_cb_t(struct IncompleteStructTy* __counted_by(size) c, int size); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

void call_consume_incomplete_cb(consume_incomplete_cb_t indirect_call) {
  // expected-error@+1{{cannot pass argument to parameter 'c' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);
  // expected-error@+1{{cannot pass argument to parameter with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb_unnamed(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);

  // expected-error@+1{{cannot pass argument to parameter with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  indirect_call(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);
}

void consume_incomplete_cb_const_count_1(struct IncompleteStructTy* __counted_by(1) c); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
void consume_incomplete_cb_unnamed_const_count_1(struct IncompleteStructTy* __counted_by(1)); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
typedef void consume_incomplete_cb_const_count_1_t(struct IncompleteStructTy* __counted_by(1) c); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}


void call_consume_incomplete_cb_const_count_1(consume_incomplete_cb_const_count_1_t indirect_call) {
  // expected-error@+1{{cannot pass argument to parameter 'c' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb_const_count_1(__unsafe_forge_single(struct IncompleteStructTy*, 0x4));
  // expected-error@+1{{cannot pass argument to parameter with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb_unnamed_const_count_1(__unsafe_forge_single(struct IncompleteStructTy*, 0x4));

  // expected-error@+1{{cannot pass argument to parameter with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  indirect_call(__unsafe_forge_single(struct IncompleteStructTy*, 0x4));
}

void consume_incomplete_cbon(struct IncompleteStructTy* __counted_by_or_null(size) c, int size); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
void consume_incomplete_cbon_unnamed(struct IncompleteStructTy* __counted_by_or_null(size), int size); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
typedef void consume_incomplete_cbon_t(struct IncompleteStructTy* __counted_by_or_null(size) c, int size); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void call_consume_incomplete_cbon(consume_incomplete_cbon_t indirect_call) {
  // expected-error@+1{{cannot pass argument to parameter 'c' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cbon(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);
  // expected-error@+1{{cannot pass argument to parameter with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cbon_unnamed(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);

  // expected-error@+1{{cannot pass argument to parameter with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  indirect_call(__unsafe_forge_single(struct IncompleteStructTy*, 0x4), 1);
}


// expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'c' with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
void wrap_consume_incomplete_cb(struct IncompleteStructTy* __counted_by(size) c, // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  int size, consume_incomplete_cb_t indirect_call) {
  // TODO: We should consider allowing this case. rdar://132031085
  //
  // This case technically doesn't require any bounds-checks because:
  //
  // 1. We assume the attribute on `c` is correct because we assume that the
  //    caller to `wrap_consume_incomplete_cb` already performed bounds-checks.
  // 2. uses of `c` and `size` in this function just pass them along to
  //    functions with the same kind of count expression. This means the bounds
  //    checks performed at the call to `wrap_consume_incomplete_cb` being true
  //    imply that all the bounds checks that would be performed here should
  //    pass. One exception to this would be if the call to
  //    `wrap_consume_incomplete_cb` was performed from non-bounds-safety code.
  //
  consume_incomplete_cb(c, size);
  consume_incomplete_cb_unnamed(c, size);
  indirect_call(c, size);
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'c' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
void wrap_consume_incomplete_cbon(struct IncompleteStructTy* __counted_by_or_null(size) c, // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int size, consume_incomplete_cbon_t indirect_call) {
  // TODO: We should consider allowing this case. rdar://132031085
  //
  // This case technically doesn't require any bounds-checks because:
  //
  // 1. We assume the attribute on `c` is correct because we assume that the
  //    caller to `wrap_consume_incomplete_cb` already performed bounds-checks.
  // 2. uses of `c` and `size` in this function just pass them along to
  //    functions with the same kind of count expression. This means the bounds
  //    checks performed at the call to `wrap_consume_incomplete_cb` being true
  //    imply that all the bounds checks that would be performed here should
  //    pass. One exception to this would be if the call to
  //    `wrap_consume_incomplete_cb` was performed from non-bounds-safety code.
  //
  consume_incomplete_cbon(c, size);
  consume_incomplete_cbon_unnamed(c, size);
  indirect_call(c, size);
}

//------------------------------------------------------------------------------
// Passing to Parameters with attributes on nested pointer
//------------------------------------------------------------------------------
void consume_incomplete_cb_nested(struct IncompleteStructTy* __counted_by(*size)* out, int* size);
void consume_incomplete_cb_unnamed_nested(struct IncompleteStructTy* __counted_by(*size)*, int* size);
typedef void consume_incomplete_cb_nested_t(struct IncompleteStructTy* __counted_by(*size)* c, int* size);

struct PtrAndCountCB {
    int size;
  struct IncompleteStructTy* __counted_by(size) ptr;
};

extern int cb_global_count;
extern struct IncompleteStructTy* __counted_by(cb_global_count) cb_global;

void call_consume_incomplete_cb_nested(consume_incomplete_cb_nested_t indirect_call, struct PtrAndCountCB* ptr_and_count) {
  // Note:
  // * Uses of `&cb_global` `&(ptr_and_count->ptr)` currently don't generate errors
  //   because only the outer most
  //   pointer is checked.
  // * Calls to the functions don't generate errors because only the outer most
  //   pointer is checked in the parameter types.
  //
  // This is ok because currently we don't checks at call sites to functions
  // with indirect __counted_by parameters. Therefore the size of
  // `struct IncompleteStructTy` isn't needed.
  consume_incomplete_cb_nested(&cb_global, &cb_global_count);
  consume_incomplete_cb_unnamed_nested(&cb_global, &cb_global_count);
  indirect_call(&cb_global, &cb_global_count);
  consume_incomplete_cb_nested(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
  consume_incomplete_cb_unnamed_nested(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
  indirect_call(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
}

void consume_incomplete_cbon_nested(struct IncompleteStructTy* __counted_by_or_null(*size)* out, int* size);
void consume_incomplete_cbon_unnamed_nested(struct IncompleteStructTy* __counted_by_or_null(*size)*, int* size);
typedef void consume_incomplete_cbon_nested_t(struct IncompleteStructTy* __counted_by_or_null(*size)* c, int* size);

extern int cbon_global_count;
extern struct IncompleteStructTy* __counted_by_or_null(cbon_global_count) cbon_global;

struct PtrAndCountCBON {
    int size;
  struct IncompleteStructTy* __counted_by_or_null(size) ptr;
};

void call_consume_incomplete_cbon_nested(consume_incomplete_cbon_nested_t indirect_call, struct PtrAndCountCBON* ptr_and_count) {
  // Note:
  // * Uses of `&cbon_global` `&(ptr_and_count->ptr)` currently don't generate errors
  //   because only the outer most
  //   pointer is checked.
  // * Calls to the functions don't generate errors because only the outer most
  //   pointer is checked in the parameter types.
  //
  // This is ok because currently we don't checks at call sites to functions
  // with indirect __counted_by parameters. Therefore the size of
  // `struct IncompleteStructTy` isn't needed.
  consume_incomplete_cbon_nested(&cbon_global, &cbon_global_count);
  consume_incomplete_cbon_unnamed_nested(&cbon_global, &cbon_global_count);
  indirect_call(&cbon_global, &cbon_global_count);
  consume_incomplete_cbon_nested(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
  consume_incomplete_cbon_unnamed_nested(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
  indirect_call(&(ptr_and_count->ptr), &(ptr_and_count->size)); // OK
}

//------------------------------------------------------------------------------
// __counted_by/__counted_by_or_null on struct members
//------------------------------------------------------------------------------

struct BuffersCBTyNotUsed {
  int count;
  struct IncompleteStructTy* __counted_by(count) buffer; // OK
  Incomplete_Struct_t* __counted_by(count) buffer_typedef; // OK
};
struct BuffersCBTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by(count) buffer; // expected-note 21{{consider using '__sized_by' instead of '__counted_by'}}
  Incomplete_Struct_t* __counted_by(count_typedef) buffer_typedef; // expected-note 21{{consider using '__sized_by' instead of '__counted_by'}}
};

void side_effect(void);

void AssignToBuffersCBTy(struct BuffersCBTy* b) {
  // Check that the diagnostic about missing assignment to the count also shows

  // expected-error@+2{{cannot assign to 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') 'b->buffer' requires corresponding assignment to 'b->count'; add self assignment 'b->count = b->count' if the value has not changed}}
  b->buffer = 0x0;
  side_effect();
  // expected-error@+2{{cannot assign to 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  // expected-error@+1{{assignment to 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') 'b->buffer_typedef' requires corresponding assignment to 'b->count_typedef'; add self assignment 'b->count_typedef = b->count_typedef' if the value has not changed}}
  b->buffer_typedef = 0x0;

  // Diagnostic about missing assignment to count should not appear.
  side_effect();
  b->count = 0;
  // expected-error@+1{{cannot assign to 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  b->buffer = 0x0;
  side_effect();
  // expected-error@+1{{cannot assign to 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  b->buffer_typedef = 0x0;
  b->count_typedef = 0;
}


struct IncompleteStructTy* ReturnBufferCBTyMember(struct BuffersCBTy* b) {
  // expected-error@+1{{cannot use 'b->buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return b->buffer;
}

Incomplete_Struct_t* ReturnBufferCBTyMemberTypeDef(struct BuffersCBTy* b) {
  // expected-error@+1{{cannot use 'b->buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  return b->buffer_typedef;
}

struct BuffersCBONTyNotUsed {
  int count;
  struct IncompleteStructTy* __counted_by_or_null(count) buffer; // OK
  Incomplete_Struct_t* __counted_by_or_null(count) buffer_typedef; // OK
};
struct BuffersCBONTy {
  int count;
  int count_typedef;
  struct IncompleteStructTy* __counted_by_or_null(count) buffer; // expected-note 21{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  Incomplete_Struct_t* __counted_by_or_null(count_typedef) buffer_typedef; // expected-note 21{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void AssignToBuffersCBONTy(struct BuffersCBONTy* b) {
  // Check that the diagnostic about missing assignment to the count also shows

  // expected-error@+2{{cannot assign to 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') 'b->buffer' requires corresponding assignment to 'b->count'; add self assignment 'b->count = b->count' if the value has not changed}}
  b->buffer = 0x0;
  side_effect();
  // expected-error@+2{{cannot assign to 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  // expected-error@+1{{assignment to 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') 'b->buffer_typedef' requires corresponding assignment to 'b->count_typedef'; add self assignment 'b->count_typedef = b->count_typedef' if the value has not changed}}
  b->buffer_typedef = 0x0;

  // Diagnostic about missing assignment to count should not appear.
  side_effect();
  b->count = 0;
  // expected-error@+1{{cannot assign to 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  b->buffer = 0x0;
  side_effect();
  // expected-error@+1{{cannot assign to 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  b->buffer_typedef = 0x0;
  b->count_typedef = 0;
}

struct IncompleteStructTy* ReturnBufferCBONTyMember(struct BuffersCBONTy* b) {
  // expected-error@+1{{cannot use 'b->buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  return b->buffer;
}

Incomplete_Struct_t* ReturnBufferCBONTyMemberTypeDef(struct BuffersCBONTy* b) {
  // expected-error@+1{{cannot use 'b->buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  return b->buffer_typedef;
}

//------------------------------------------------------------------------------
// Initialization of struct members with counted_by/counted_by_or_null
//------------------------------------------------------------------------------

// TODO: We should consider allowing implicit and explicit zero initialization
// of __counted_by_or_null pointers. rdar://129424354

struct BufferCBNonZeroConstCountTy {
  int extra_field;
  struct IncompleteStructTy* __counted_by(1) ptr; // expected-note 4{{consider using '__sized_by' instead of '__counted_by'}}
};

struct BufferCBNonZeroConstCountFlippedFieldOrderTy {
  struct IncompleteStructTy* __counted_by(1) ptr; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  int extra_field;
};

struct BufferCBNonZeroDynCountTy {
  unsigned int count;
  struct IncompleteStructTy* __counted_by(count+1) ptr; // expected-note 5{{consider using '__sized_by' instead of '__counted_by'}}
};

union BufferCBOrOther {
  struct BuffersCBTy buf;
  int other;
};

void InitBuffersCBTy(int size) {
  // Designated initializers
  struct BuffersCBTy desig_init_0 = {
    .count = size,
    .count_typedef = size,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    .buffer = 0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    .buffer_typedef = 0x0
  };

  struct BuffersCBTy desig_init_1 = {
    // .count and .count_typedef not explicitly initialized but are implicitly zero initialized
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    .buffer = 0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    .buffer_typedef = 0x0
  };

  struct BuffersCBTy desig_init_partial = {
    .count = size,
    .count_typedef = size,
    // .buffer and .buffer_typedef are not explicit initialized but are implicitly zero initialized
  };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  struct BuffersCBTy implicit_all_zero_init = {0}; // Implicit field init
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // non-designated initializer
  struct BuffersCBTy non_design_init_0 = {
    0,
    0,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    0x0
  };

  struct BuffersCBTy non_design_init_1 = { 0, 0 };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  struct BuffersCBTy desig_init_invalid_count = {
    .count = 1,
    // expected-error@+2{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'desig_init_invalid_count.buffer' of type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    .buffer = 0x0
  };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  struct BuffersCBTy desig_init_invalid_count_partial = {
    .count = 1
  };
  // expected-error@-1{{implicitly initializing 'desig_init_invalid_count_partial.buffer' of type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-3{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  struct BuffersCBTy non_desig_init_invalid_count = {
    1,
    0,
    // expected-error@+2{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'non_desig_init_invalid_count.buffer' of type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    0x0
   };

 
  struct BuffersCBTy non_desig_init_invalid_count_partial = {1};
  // expected-error@-1{{implicitly initializing 'non_desig_init_invalid_count_partial.buffer' of type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-3{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // Cases where zero-init would create an invalid count
  struct BufferCBNonZeroConstCountTy design_init_const_count = {
    // expected-error@+2{{cannot initialize 'BufferCBNonZeroConstCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'design_init_const_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    .ptr = 0x0,
    .extra_field = 0
  };

  struct BufferCBNonZeroConstCountTy design_init_const_count_partial_explicit = {
    // .ptr is implicitly zero initialized
    .extra_field = 0x0
  };
  // expected-error@-1{{implicitly initializing 'design_init_const_count_partial_explicit.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BufferCBNonZeroConstCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  struct BufferCBNonZeroConstCountTy implicit_all_zero_init_const_count = {0};
  // expected-error@-1{{implicitly initializing 'implicit_all_zero_init_const_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BufferCBNonZeroConstCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  // When the ptr comes first it's seen as an explicit assignment when we write ` = {0}` so we get the incomplete pointee type error diagnostic
  // expected-error@+2{{cannot initialize 'BufferCBNonZeroConstCountFlippedFieldOrderTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{initializing 'implicit_all_zero_init_const_count_ptr_init.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  struct BufferCBNonZeroConstCountFlippedFieldOrderTy implicit_all_zero_init_const_count_ptr_init = {0};

  struct BufferCBNonZeroDynCountTy design_init_non_zero_dyn_count = {
    .count = 0x0,
    // expected-error@+2{{cannot initialize 'BufferCBNonZeroDynCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'design_init_non_zero_dyn_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    .ptr = 0x0
  };

  struct BufferCBNonZeroDynCountTy design_init_non_zero_dyn_count_partial_init = {
    // count is implicitly zero
    // expected-error@+2{{cannot initialize 'BufferCBNonZeroDynCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'design_init_non_zero_dyn_count_partial_init.ptr' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    .ptr = 0x0
  };

  struct BufferCBNonZeroDynCountTy design_init_non_zero_dyn_count_partial_init2 = {
    // ptr is implicitly zero initialized
    .count = 0x0
  };
  // expected-error@-1{{implicitly initializing 'design_init_non_zero_dyn_count_partial_init2.ptr' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BufferCBNonZeroDynCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  struct BufferCBNonZeroDynCountTy implicit_all_zero_init_non_zero_dyn_count = {0};
  // expected-error@-1{{implicitly initializing 'implicit_all_zero_init_non_zero_dyn_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  // expected-error@-2{{cannot implicitly initialize 'BufferCBNonZeroDynCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  struct BufferCBNonZeroDynCountTy non_desig_init_non_zero_dyn_count = {
    0,
    // expected-error@+2{{cannot initialize 'BufferCBNonZeroDynCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    // expected-error@+1{{initializing 'non_desig_init_non_zero_dyn_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(count + 1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
    0
  };

  // Struct inside a union
  union BufferCBOrOther UnionDesignInitOther = {.other = 0x0 };
  union BufferCBOrOther UnionZeroInit = {0};
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  union BufferCBOrOther UnionDesignInitBufZeroInitStructFields = {.buf = {0}};
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  union BufferCBOrOther UnionDesignInitBufDesignInitStructFields = {.buf = {.count = 0, .buffer = 0x0}};
  // expected-error@-1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
}

struct BuffersCBTy GlobalBuffersCBTy_design_init =  {
  .count = 0,
  .count_typedef = 0,
  // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  .buffer = 0x0,
  // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  .buffer_typedef = 0
};

struct BuffersCBTy GlobalBuffersCBTy_non_design_init = {
  0,
  0,
  // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  0,
  // expected-error@+1{{cannot initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  0
};

struct BuffersCBTy GlobalBuffersCBTy_design_init_partial =  {
  .count = 0,
  .count_typedef = 0
  // buffer and buffer_typedef are implicitly zero initialized
};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

struct BuffersCBTy GlobalBuffersCBTy_non_design_init_partial = {
  0,
  0,
  // buffer and buffer_typedef are implicitly zero initialized
};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

struct BuffersCBTy GlobalBuffersCBTy_all_zero_init = {0};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBTy::buffer' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBTy::buffer_typedef' with '__counted_by' attributed type 'Incomplete_Struct_t *__single __counted_by(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}


// expected-error@+2{{cannot initialize 'BufferCBNonZeroConstCountFlippedFieldOrderTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@+1{{initializing 'GlobalBuffersCBTy_implicit_all_zero_init_const_count_ptr_init.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
struct BufferCBNonZeroConstCountFlippedFieldOrderTy GlobalBuffersCBTy_implicit_all_zero_init_const_count_ptr_init = {0};

struct BufferCBNonZeroConstCountTy  GlobalBuffersCBTy_const_non_zero_count = {
  // expected-error@+2{{cannot initialize 'BufferCBNonZeroConstCountTy::ptr' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{initializing 'GlobalBuffersCBTy_const_non_zero_count.ptr' of type 'struct IncompleteStructTy *__single __counted_by(1)' (aka 'struct IncompleteStructTy *__single') and count value of 1 with null always fails}}
  .ptr = 0x0
};

// counted_by_or_null variants


union BufferCBONOrOther {
  struct BuffersCBONTy buf;
  int other;
};

struct BufferCBONNonZeroConstCountTy {
  int extra_field;
  struct IncompleteStructTy* __counted_by_or_null(1) ptr; // expected-note 4{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

struct BufferCBONNonZeroConstCountFlippedFieldOrderTy {
  struct IncompleteStructTy* __counted_by_or_null(1) ptr; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  int extra_field;
};

struct BufferCBONNonZeroDynCountTy {
  unsigned int count;
  struct IncompleteStructTy* __counted_by_or_null(count+1) ptr; // expected-note 5{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void InitBuffersCBONTy(int size) {
  // Designated initializers
  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy desig_init_0 = {
    .count = size,
    .count_typedef = size,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    .buffer = 0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    .buffer_typedef = 0x0
  };

   // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy desig_init_1 = {
    // .count and .count_typedef not explicitly initialized but are implicitly zero initialized
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    .buffer = 0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    .buffer_typedef = 0x0
  };

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy desig_init_partial = {
    .count = size,
    .count_typedef = size,
    // .buffer and .buffer_typedef are not explicit initialized but are implicitly zero initialized
  };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy implicit_all_zero_init = {0}; // Implicit field init
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // non-designated initializer
  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy non_design_init_0 = {
    0,
    0,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    0x0
  };

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy non_design_init_1 = { 0, 0 };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // TODO: Explicit and implicit  0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy desig_init_invalid_count = {
    .count = 1,
    // expected-error@+2{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    .buffer = 0x0
  };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy desig_init_explicit_non_zero_count_partial = {
    .count = 1
  };
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy non_desig_init_invalid_count = {
    1,
    0,
    // expected-error@+2{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    0x0,
    // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
    0x0
   };

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BuffersCBONTy non_desig_init_non_zerocount_partial = {1};
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  // Cases where zero-init would create an invalid count
  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroConstCountTy design_init_const_count = {
    // expected-error@+2{{cannot initialize 'BufferCBONNonZeroConstCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    .ptr = 0x0,
    .extra_field = 0
  };

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroConstCountTy design_init_const_count_partial_explicit = {
    // .ptr is implicitly zero initialized
    .extra_field = 0x0
  };
  // expected-error@-1{{cannot implicitly initialize 'BufferCBONNonZeroConstCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroConstCountTy implicit_all_zero_init_const_count = {0};
  // expected-error@-1{{cannot implicitly initialize 'BufferCBONNonZeroConstCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  // When the ptr comes first it's seen as an explicit assignment when we write ` = {0}` so we get the incomplete pointee type error diagnostic
  // expected-error@+2{{cannot initialize 'BufferCBONNonZeroConstCountFlippedFieldOrderTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  //
  struct BufferCBONNonZeroConstCountFlippedFieldOrderTy implicit_all_zero_init_const_count_ptr_init = {0};

  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroDynCountTy design_init_non_zero_dyn_count = {
    .count = 0x0,
    // expected-error@+2{{cannot initialize 'BufferCBONNonZeroDynCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    .ptr = 0x0
  };

  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroDynCountTy design_init_non_zero_dyn_count_partial_init = {
    // count is implicitly zero
    // expected-error@+2{{cannot initialize 'BufferCBONNonZeroDynCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    .ptr = 0x0
  };

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroDynCountTy design_init_non_zero_dyn_count_partial_init2 = {
    // ptr is implicitly zero initialized
    .count = 0x0
  };
  // expected-error@-1{{cannot implicitly initialize 'BufferCBONNonZeroDynCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  // TODO: Implicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroDynCountTy implicit_all_zero_init_non_zero_dyn_count = {0};
  // expected-error@-1{{cannot implicitly initialize 'BufferCBONNonZeroDynCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}

  // TODO: Explicit 0x0 initialization should be allowed. rdar://129424354
  struct BufferCBONNonZeroDynCountTy non_desig_init_non_zero_dyn_count = {
    0,
    // expected-error@+2{{cannot initialize 'BufferCBONNonZeroDynCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count + 1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
    //
    0
  };

   // Struct inside a union
  union BufferCBONOrOther UnionDesignInitOther = {.other = 0x0 };
  union BufferCBONOrOther UnionZeroInit = {0};
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  union BufferCBONOrOther UnionDesignInitBufZeroInitStructFields = {.buf = {0}};
  // expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

  union BufferCBONOrOther UnionDesignInitBufDesignInitStructFields = {.buf = {.count = 0, .buffer = 0x0}};
  // expected-error@-1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
}

// XXX

struct BuffersCBONTy GlobalBuffersCBONTy_design_init =  {
  .count = 0,
  .count_typedef = 0,
  // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  .buffer = 0x0,
  // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  .buffer_typedef = 0
};

struct BuffersCBONTy GlobalBuffersCBONTy_non_design_init = {
  0,
  0,
  // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  0,
  // expected-error@+1{{cannot initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
  0
};

struct BuffersCBONTy GlobalBuffersCBONTy_design_init_partial =  {
  .count = 0,
  .count_typedef = 0
  // buffer and buffer_typedef are implicitly zero initialized
};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

struct BuffersCBONTy GlobalBuffersCBONTy_non_design_init_partial = {
  0,
  0,
  // buffer and buffer_typedef are implicitly zero initialized
};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}

struct BuffersCBONTy GlobalBuffersCBONTy_all_zero_init = {0};
// expected-error@-1{{cannot implicitly initialize 'BuffersCBONTy::buffer' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@-2{{cannot implicitly initialize 'BuffersCBONTy::buffer_typedef' with '__counted_by_or_null' attributed type 'Incomplete_Struct_t *__single __counted_by_or_null(count_typedef)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}


// expected-error@+2{{cannot initialize 'BufferCBONNonZeroConstCountFlippedFieldOrderTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
//
struct BufferCBONNonZeroConstCountFlippedFieldOrderTy GlobalBuffersCBONTy_implicit_all_zero_init_const_count_ptr_init = {0};

struct BufferCBONNonZeroConstCountTy  GlobalBuffersCBONTy_const_non_zero_count = {
  // expected-error@+2{{cannot initialize 'BufferCBONNonZeroConstCountTy::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(1)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  //
  .ptr = 0x0
};

//------------------------------------------------------------------------------
// Local __counted_by variables
//------------------------------------------------------------------------------
void local_cb_init_and_assign(int s) {
  int size = s;

  // expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'local_init' with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(size) local_init = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  local_init = 0x0;

  int implicit_size = s;
  // expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'implicit_init' with type 'struct IncompleteStructTy *__single __counted_by(implicit_size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(implicit_size) implicit_init; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  implicit_init = 0x0;
}

void local_cb_init_and_assign_constant_count(void) {
  // Check we also emit diagnostics about assigning nullptr to `__counted_by(X)` where X > 0
  //
  // expected-error@+2{{cannot apply '__counted_by' attribute to variable definition 'local_init' with type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{initializing 'local_init' of type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') and count value of 5 with null always fails}}
  struct IncompleteStructTy* __counted_by(5) local_init = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  local_init = 0x0; // Diagnostic suppressed because the VarDecl is invalid

  // There should be no diagnostic about assigning nullptr
  // TODO: We should consider allowing this given that the type size isn't
  // really needed when the count is 0 (rdar://129424147).
  // expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'local_init_zero' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(0) local_init_zero = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  local_init_zero = 0x0; // Diagnostic suppressed because the VarDecl is invalid

  // expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'local_init2' with type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by(5) local_init2 = __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x4, 4); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  local_init2 = __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x4, 4); // Diagnostic suppressed because the VarDecl is invalid
}

void local_cbon_init_and_assign(int s) {
  int size = s;

  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'local_init' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(size) local_init = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  local_init = 0x0;

  int implicit_size = s;
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'implicit_init' with type 'struct IncompleteStructTy *__single __counted_by_or_null(implicit_size)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(implicit_size) implicit_init; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  implicit_init = 0x0;
}

void local_cbon_init_and_assign_constant_count(void) {
  // TODO: We should consider allowing this because the assignment of nullptr
  // means the type size isn't needed (rdar://129424354).
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'local_init' with type 'struct IncompleteStructTy *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(5) local_init = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  local_init = 0x0; // Diagnostic suppressed because the VarDecl is invalid

  // There should be no diagnostic about assigning nullptr
  // TODO: We should consider allowing this given that the type size isn't
  // really needed when the count is 0 (rdar://129424147).
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'local_init_zero' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(0) local_init_zero = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  local_init_zero = 0x0; // Diagnostic suppressed because the VarDecl is invalid

  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'local_init2' with type 'struct IncompleteStructTy *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  struct IncompleteStructTy* __counted_by_or_null(5) local_init2 = __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x4, 4); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  local_init2 = __unsafe_forge_bidi_indexable(struct IncompleteStructTy*, 0x4, 4); // Diagnostic suppressed because the VarDecl is invalid
}

//------------------------------------------------------------------------------
// Global __counted_by variables
//------------------------------------------------------------------------------
// NOTE: Tentative definitions are mostly tested in `counted_by_type_incomplete_completable_struct_tentative_defs.c`.

extern int external_count;
// expected-note@+1 3{{consider using '__sized_by' instead of '__counted_by'}}
extern struct IncompleteStructTy* __counted_by(external_count) GlobalCBPtrToIncompleteTy; // OK
extern Incomplete_Struct_t* __counted_by(external_count) GlobalCBPtrToIncompleteTyTypeDef; // OK

void use_GlobalCBPtrToIncompleteTy(void) {
  // expected-error@+2{{cannot assign to 'GlobalCBPtrToIncompleteTy' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(external_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by(external_count)' (aka 'struct IncompleteStructTy *__single') 'GlobalCBPtrToIncompleteTy' requires corresponding assignment to 'external_count'; add self assignment 'external_count = external_count' if the value has not changed}}
  GlobalCBPtrToIncompleteTy = 0x0;
  // expected-error@+1{{cannot use 'GlobalCBPtrToIncompleteTy' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(external_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  GlobalCBPtrToIncompleteTy[0] = 0;
  // expected-error@+1{{cannot use 'GlobalCBPtrToIncompleteTy' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(external_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb(GlobalCBPtrToIncompleteTy, external_count);
}

static int global_count;
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInit' with type 'struct IncompleteStructTy *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
static struct IncompleteStructTy* __counted_by(global_count) GlobalCBPtrImplicitInit; // expected-note 4{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInitTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
static Incomplete_Struct_t* __counted_by(global_count) GlobalCBPtrImplicitInitTypeDef; // expected-note {{consider using '__sized_by' instead of '__counted_by'}}

void use_GlobalCBPtrImplicitInit(void) {
  // expected-error@+2{{cannot assign to 'GlobalCBPtrImplicitInit' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') 'GlobalCBPtrImplicitInit' requires corresponding assignment to 'global_count'; add self assignment 'global_count = global_count' if the value has not changed}}
  GlobalCBPtrImplicitInit = 0x0;
  // expected-error@+1{{cannot use 'GlobalCBPtrImplicitInit' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  GlobalCBPtrImplicitInit[0] = 0;
  // expected-error@+1{{cannot use 'GlobalCBPtrImplicitInit' with '__counted_by' attributed type 'struct IncompleteStructTy *__single __counted_by(global_count)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cb(GlobalCBPtrImplicitInit, global_count);
}

int global_count_non_static = 0;
// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInit' with type 'struct IncompleteStructTy *__single __counted_by(global_count_non_static)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(global_count_non_static) GlobalCBPtrExplicitInit = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(global_count_non_static)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by(global_count_non_static) GlobalCBPtrExplicitInitTypeDef = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

void use_GlobalCBPtrExplicitInit(void) {
  // No diagnostics because the VarDecl is marked as invalid at this point
  GlobalCBPtrExplicitInit = 0x0;
  GlobalCBPtrExplicitInit[0] = 0;
  consume_incomplete_cb(GlobalCBPtrExplicitInit, global_count_non_static);
}

// This is very unidiomatic C but it seems to be legal.
// expected-warning@+1{{'extern' variable has an initializer}}
extern int global_count_extern = 0;
// expected-error@+2{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitExtern' with type 'struct IncompleteStructTy *__single __counted_by(global_count_extern)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-warning@+1 2{{'extern' variable has an initializer}} TODO: This shouldn't be emitted twice. rdar://133001618
extern struct IncompleteStructTy* __counted_by(global_count_extern) GlobalCBPtrExplicitInitExtern = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+2{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitTypeDefExtern' with type 'Incomplete_Struct_t *__single __counted_by(global_count_extern)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
// expected-warning@+1 2{{'extern' variable has an initializer}} TODO: This shouldn't be emitted twice. rdar://133001618
extern Incomplete_Struct_t* __counted_by(global_count_extern) GlobalCBPtrExplicitInitTypeDefExtern = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

// TODO: We should consider allowing this given that the pointee type size isn't
// really needed when the count is 0 (rdar://129424147)
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInitConstantZeroCount' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(0) GlobalCBPtrImplicitInitConstantZeroCount; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInitConstantZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by(0) GlobalCBPtrImplicitInitConstantZeroCountTypeDef; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitConstantZeroCount' with type 'struct IncompleteStructTy *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by(0) GlobalCBPtrExplicitInitConstantZeroCount = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitConstantZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by(0) GlobalCBPtrExplicitInitConstantZeroCountTypeDef = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

// expected-error@+2{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInitConstantNonZeroCount' with type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@+1{{implicitly initializing 'GlobalCBPtrImplicitInitConstantNonZeroCount' of type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') and count value of 5 with null always fails}}
struct IncompleteStructTy* __counted_by(5) GlobalCBPtrImplicitInitConstantNonZeroCount; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+2{{cannot apply '__counted_by' attribute to tentative variable definition 'GlobalCBPtrImplicitInitConstantNonZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
// expected-error@+1{{implicitly initializing 'GlobalCBPtrImplicitInitConstantNonZeroCountTypeDef' of type 'Incomplete_Struct_t *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') and count value of 5 with null always fails}}
Incomplete_Struct_t* __counted_by(5) GlobalCBPtrImplicitInitConstantNonZeroCountTypeDef; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+2{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitConstantNonZeroCount' with type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-error@+1{{initializing 'GlobalCBPtrExplicitInitConstantNonZeroCount' of type 'struct IncompleteStructTy *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') and count value of 5 with null always fails}}
struct IncompleteStructTy* __counted_by(5) GlobalCBPtrExplicitInitConstantNonZeroCount = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
// expected-error@+2{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExplicitInitConstantNonZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
// expected-error@+1{{initializing 'GlobalCBPtrExplicitInitConstantNonZeroCountTypeDef' of type 'Incomplete_Struct_t *__single __counted_by(5)' (aka 'struct IncompleteStructTy *__single') and count value of 5 with null always fails}}
Incomplete_Struct_t* __counted_by(5) GlobalCBPtrExplicitInitConstantNonZeroCountTypeDef = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

//------------------------------------------------------------------------------
// Global __counted_by_or_null variables
//------------------------------------------------------------------------------
// NOTE: Tentative definitions are mostly tested in `counted_by_type_incomplete_completable_struct_tentative_defs.c`.

extern int external_count_cbon;
extern int external_count_cbon_typedef;
// expected-note@+1 3{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
extern struct IncompleteStructTy* __counted_by_or_null(external_count_cbon) GlobalCBONPtrToIncompleteTy; // OK
extern Incomplete_Struct_t* __counted_by_or_null(external_count_cbon_typedef) GlobalCBONPtrToIncompleteTyTypeDef; // OK

void use_GlobalCBONPtrToIncompleteTy(void) {
  // expected-error@+2{{cannot assign to 'GlobalCBONPtrToIncompleteTy' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(external_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by_or_null(external_count_cbon)' (aka 'struct IncompleteStructTy *__single') 'GlobalCBONPtrToIncompleteTy' requires corresponding assignment to 'external_count_cbon'; add self assignment 'external_count_cbon = external_count_cbon' if the value has not changed}}
  GlobalCBONPtrToIncompleteTy = 0x0;
  // expected-error@+1{{cannot use 'GlobalCBONPtrToIncompleteTy' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(external_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  GlobalCBONPtrToIncompleteTy[0] = 0;
  // expected-error@+1{{cannot use 'GlobalCBONPtrToIncompleteTy' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(external_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cbon(GlobalCBONPtrToIncompleteTy, external_count);
}



static int global_count_cbon;
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInit' with type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
static struct IncompleteStructTy* __counted_by_or_null(global_count_cbon) GlobalCBONPtrImplicitInit; // expected-note 4{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInitTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
static Incomplete_Struct_t* __counted_by_or_null(global_count_cbon) GlobalCBONPtrImplicitInitTypeDef; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}



void use_GlobalCBONPtrImplicitInit(void) {
  // expected-error@+2{{cannot assign to 'GlobalCBONPtrImplicitInit' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-error@+1{{assignment to 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') 'GlobalCBONPtrImplicitInit' requires corresponding assignment to 'global_count_cbon'; add self assignment 'global_count_cbon = global_count_cbon' if the value has not changed}}
  GlobalCBONPtrImplicitInit = 0x0;
  // expected-error@+1{{cannot use 'GlobalCBONPtrImplicitInit' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  GlobalCBONPtrImplicitInit[0] = 0;
  // expected-error@+1{{cannot use 'GlobalCBONPtrImplicitInit' with '__counted_by_or_null' attributed type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
  consume_incomplete_cbon(GlobalCBONPtrImplicitInit, global_count_cbon);
}



int global_count_cbon_non_static = 0;
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInit' with type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon_non_static)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(global_count_cbon_non_static) GlobalCBONPtrExplicitInit = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(global_count_cbon_non_static)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by_or_null(global_count_cbon_non_static) GlobalCBONPtrExplicitInitTypeDef = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

void use_GlobalCBONPtrExplicitInit(void) {
  // No diagnostics because the VarDecl is marked as invalid at this point
  GlobalCBONPtrExplicitInit = 0x0;
  GlobalCBONPtrExplicitInit[0] = 0;
  consume_incomplete_cbon(GlobalCBONPtrExplicitInit, global_count_cbon_non_static);
}



// This is very unidiomatic C but it seems to be legal.
// expected-warning@+1{{'extern' variable has an initializer}}
extern int global_count_cbon_extern = 0;
// expected-error@+2{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitExtern' with type 'struct IncompleteStructTy *__single __counted_by_or_null(global_count_cbon_extern)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
// expected-warning@+1 2{{'extern' variable has an initializer}} TODO: This shouldn't be emitted twice. rdar://133001618
extern struct IncompleteStructTy* __counted_by_or_null(global_count_cbon_extern) GlobalCBONPtrExplicitInitExtern = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+2{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitTypeDefExtern' with type 'Incomplete_Struct_t *__single __counted_by_or_null(global_count_cbon_extern)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
// expected-warning@+1 2{{'extern' variable has an initializer}} TODO: This shouldn't be emitted twice. rdar://133001618
extern Incomplete_Struct_t* __counted_by_or_null(global_count_cbon_extern) GlobalCBONPtrExplicitInitTypeDefExtern = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}


// TODO: We should consider allowing this given that the pointee type size isn't
// really needed when the count is 0 (rdar://129424147)
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInitConstantZeroCount' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(0) GlobalCBONPtrImplicitInitConstantZeroCount; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInitConstantZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by_or_null(0) GlobalCBONPtrImplicitInitConstantZeroCountTypeDef; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitConstantZeroCount' with type 'struct IncompleteStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(0) GlobalCBONPtrExplicitInitConstantZeroCount = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitConstantZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(0)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by_or_null(0) GlobalCBONPtrExplicitInitConstantZeroCountTypeDef = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

// Unlike `__counted_by` assigning 0x0 (implicitly or explicitly) is allowed for `__counted_by_or_null`
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInitConstantNonZeroCount' with type 'struct IncompleteStructTy *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(5) GlobalCBONPtrImplicitInitConstantNonZeroCount; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to tentative variable definition 'GlobalCBONPtrImplicitInitConstantNonZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by_or_null(5) GlobalCBONPtrImplicitInitConstantNonZeroCountTypeDef; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitConstantNonZeroCount' with type 'struct IncompleteStructTy *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'struct IncompleteStructTy' is incomplete}}
struct IncompleteStructTy* __counted_by_or_null(5) GlobalCBONPtrExplicitInitConstantNonZeroCount = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExplicitInitConstantNonZeroCountTypeDef' with type 'Incomplete_Struct_t *__single __counted_by_or_null(5)' (aka 'struct IncompleteStructTy *__single') because the pointee type 'Incomplete_Struct_t' (aka 'struct IncompleteStructTy') is incomplete}}
Incomplete_Struct_t* __counted_by_or_null(5) GlobalCBONPtrExplicitInitConstantNonZeroCountTypeDef = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

//------------------------------------------------------------------------------
// No explicit forward decl
//------------------------------------------------------------------------------

// expected-note@+1 2{{consider providing a complete definition for 'NoExplicitForwardDecl_t' (aka 'struct NoExplicitForwardDecl')}}
typedef struct NoExplicitForwardDecl NoExplicitForwardDecl_t;

extern NoExplicitForwardDecl_t* __counted_by(0) NoExplicitForwardDeclGlobalCBPtr; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
void consume_NoExplicitForwardDeclGlobalCBPtr(void) {
  // expected-error@+1{{cannot assign to 'NoExplicitForwardDeclGlobalCBPtr' with '__counted_by' attributed type 'NoExplicitForwardDecl_t *__single __counted_by(0)' (aka 'struct NoExplicitForwardDecl *__single') because the pointee type 'NoExplicitForwardDecl_t' (aka 'struct NoExplicitForwardDecl') is incomplete}}
  NoExplicitForwardDeclGlobalCBPtr = 0x0;
}

extern NoExplicitForwardDecl_t* __counted_by_or_null(0) NoExplicitForwardDeclGlobalCBONPtr; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
void consume_NoExplicitForwardDeclGlobalCBONPtr(void) {
  // expected-error@+1{{cannot assign to 'NoExplicitForwardDeclGlobalCBONPtr' with '__counted_by_or_null' attributed type 'NoExplicitForwardDecl_t *__single __counted_by_or_null(0)' (aka 'struct NoExplicitForwardDecl *__single') because the pointee type 'NoExplicitForwardDecl_t' (aka 'struct NoExplicitForwardDecl') is incomplete}}
  NoExplicitForwardDeclGlobalCBONPtr = 0x0;
}

//------------------------------------------------------------------------------
// Array element initialization
//
// Currently this appears to be forbidden
//------------------------------------------------------------------------------

void array_elts_init_cb(void) {
  int size;
  // expected-error@+1{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
  struct IncompleteStructTy*__counted_by(size) arr[2] = { 0x0, 0x0 };
}

void array_elts_init_cbon(void) {
  int size;
  // expected-error@+1{{'__counted_by_or_null' attribute on nested pointer type is only allowed on indirect parameters}}
  struct IncompleteStructTy*__counted_by_or_null(size) arr[2] = { 0x0, 0x0 };
}

//------------------------------------------------------------------------------
// Casting
//------------------------------------------------------------------------------
// TODO: These should cause errors to be emitted.
// rdar://131621712
void explicit_cast_cb_to_single(struct IncompleteStructTy* p) {
  struct IncompleteStructTy* __single tmp =
    (struct IncompleteStructTy* __counted_by(1)) p;
}

void explicit_cast_cbon_to_single(struct IncompleteStructTy* p) {
  struct IncompleteStructTy* __single tmp =
    (struct IncompleteStructTy* __counted_by_or_null(1)) p;
}


void explicit_cast_cb_to_bidi(struct IncompleteStructTy* p) {
  // TODO: This diagnostic is misleading. It says __single but it should probably be `__counted_by(2)`. rdar://133002045
  // expected-error@+1{{cannot initialize indexable pointer with type 'struct IncompleteStructTy *__bidi_indexable' from __single pointer to incomplete type 'struct IncompleteStructTy *__single'; consider declaring pointer 'local_bidi' as '__single'}}
  struct IncompleteStructTy* local_bidi = (struct IncompleteStructTy* __counted_by(2)) p; // expected-note{{pointer 'local_bidi' declared here}}
}

void explicit_cast_cbon_to_bidi(struct IncompleteStructTy* p) {
  // TODO: This diagnostic is misleading. It says __single but it should probably be `__counted_by(2)`. rdar://133002045
  // expected-error@+1{{cannot initialize indexable pointer with type 'struct IncompleteStructTy *__bidi_indexable' from __single pointer to incomplete type 'struct IncompleteStructTy *__single'; consider declaring pointer 'local_bidi' as '__single'}}
  struct IncompleteStructTy* local_bidi = (struct IncompleteStructTy* __counted_by_or_null(2)) p; // expected-note{{pointer 'local_bidi' declared here}}
}

//------------------------------------------------------------------------------
// Completing the pointee type allows usage
//------------------------------------------------------------------------------

// expected-note@+1 20{{consider providing a complete definition for 'struct IncompleteLaterCompletedStructTy'}}
struct IncompleteLaterCompletedStructTy;

// Confirm using the type is an error at this point
// expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'GlobalCBPtrExpectErr' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
struct IncompleteLaterCompletedStructTy*__counted_by(0) GlobalCBPtrExpectErr = 0x0; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
struct IncompleteLaterCompletedStructTy*__counted_by(0) GlobalCBPtrTentativeDefUseWillErr; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'GlobalCBONPtrExpectErr' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) GlobalCBONPtrExpectErr = 0x0; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) GlobalCBONPtrTentativeDefUseWillErr; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

struct StructPtrIncompleteLaterCompleted {
  int count;
  struct IncompleteLaterCompletedStructTy*__counted_by(count) ptr; // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
};
struct StructCBONPtrIncompleteLaterCompleted {
  int count;
  struct IncompleteLaterCompletedStructTy*__counted_by_or_null(count) ptr; // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void consume_IncompleteLaterCompletedStructTy(struct IncompleteLaterCompletedStructTy*__counted_by(0) p); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
void consume_CBON_IncompleteLaterCompletedStructTy(struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) p); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

struct IncompleteLaterCompletedStructTy*__counted_by(0) ret_IncompleteLaterCompletedStructTy(void); // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) ret_CBON_IncompleteLaterCompletedStructTy(void); // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

// expected-error@+1{{cannot apply '__counted_by' attribute to return type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') on a function definition because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
struct IncompleteLaterCompletedStructTy*__counted_by(0) test_cb_expect_err( // expected-note 2{{consider using '__sized_by' instead of '__counted_by'}}
  // expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'param' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') on a function definition because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct IncompleteLaterCompletedStructTy*__counted_by(0) param) { // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  // expected-error@+1{{cannot apply '__counted_by' attribute to variable definition 'local' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct IncompleteLaterCompletedStructTy*__counted_by(0) local; // expected-note{{consider using '__sized_by' instead of '__counted_by'}}

  // expected-error@+1{{cannot assign to 'GlobalCBPtrTentativeDefUseWillErr' with '__counted_by' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  GlobalCBPtrTentativeDefUseWillErr = 0;

  // expected-error@+1{{cannot initialize 'StructPtrIncompleteLaterCompleted::ptr' with '__counted_by' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(count)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct StructPtrIncompleteLaterCompleted tmp = { .count = 0, .ptr = 0x0 };

  struct StructPtrIncompleteLaterCompleted tmp2;
  // expected-error@+1{{cannot use 'tmp2.ptr' with '__counted_by' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(count)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  consume_IncompleteLaterCompletedStructTy(tmp2.ptr);

  // expected-error@+1{{cannot pass argument to parameter 'p' with '__counted_by' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  consume_IncompleteLaterCompletedStructTy(0x0);

  // expected-error@+1{{cannot call 'ret_IncompleteLaterCompletedStructTy' with '__counted_by' attributed return type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  ret_IncompleteLaterCompletedStructTy();

  // expected-error@+1{{cannot return '__counted_by' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  return 0x0;
}

// expected-error@+1{{cannot apply '__counted_by_or_null' attribute to return type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') on a function definition because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) test_cbon_expect_err( // expected-note 2{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'param' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') on a function definition because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) param) { // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to variable definition 'local' with type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) local; // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}

  // expected-error@+1{{cannot assign to 'GlobalCBONPtrTentativeDefUseWillErr' with '__counted_by_or_null' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  GlobalCBONPtrTentativeDefUseWillErr = 0;

  // expected-error@+1{{cannot initialize 'StructCBONPtrIncompleteLaterCompleted::ptr' with '__counted_by_or_null' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  struct StructCBONPtrIncompleteLaterCompleted tmp = { .count = 0, .ptr = 0x0 };

  struct StructCBONPtrIncompleteLaterCompleted tmp2;
  // expected-error@+1{{cannot use 'tmp2.ptr' with '__counted_by_or_null' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(count)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  consume_CBON_IncompleteLaterCompletedStructTy(tmp2.ptr);

  // expected-error@+1{{cannot pass argument to parameter 'p' with '__counted_by_or_null' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  consume_CBON_IncompleteLaterCompletedStructTy(0x0);

  // expected-error@+1{{cannot call 'ret_CBON_IncompleteLaterCompletedStructTy' with '__counted_by_or_null' attributed return type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  ret_CBON_IncompleteLaterCompletedStructTy();

  // expected-error@+1{{cannot return '__counted_by_or_null' attributed type 'struct IncompleteLaterCompletedStructTy *__single __counted_by_or_null(0)' (aka 'struct IncompleteLaterCompletedStructTy *__single') because the pointee type 'struct IncompleteLaterCompletedStructTy' is incomplete}}
  return 0x0;
}


// Now complete the type and confirm it can be used
struct IncompleteLaterCompletedStructTy {
  int field;
};

struct IncompleteLaterCompletedStructTy*__counted_by(0) GlobalCBPtrExpectNoErr = 0x0;

struct IncompleteLaterCompletedStructTy*__counted_by(0) test_cb_expect_no_err(
  struct IncompleteLaterCompletedStructTy*__counted_by(0) param) {
  struct IncompleteLaterCompletedStructTy*__counted_by(0) local;

  GlobalCBPtrTentativeDefUseWillErr = 0;

  struct StructPtrIncompleteLaterCompleted tmp = { .count = 0, .ptr = 0x0 };
  struct StructPtrIncompleteLaterCompleted tmp2;
  consume_IncompleteLaterCompletedStructTy(tmp2.ptr);

  consume_IncompleteLaterCompletedStructTy(0x0);
  ret_IncompleteLaterCompletedStructTy();

  return 0x0;
}

struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) test_cbon_expect_no_err(
  struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) param) {
  struct IncompleteLaterCompletedStructTy*__counted_by_or_null(0) local;

  GlobalCBONPtrTentativeDefUseWillErr = 0;

  struct StructCBONPtrIncompleteLaterCompleted tmp = { .count = 0, .ptr = 0x0 };
  struct StructCBONPtrIncompleteLaterCompleted tmp2;
  consume_CBON_IncompleteLaterCompletedStructTy(tmp2.ptr);

  consume_CBON_IncompleteLaterCompletedStructTy(0x0);
  ret_CBON_IncompleteLaterCompletedStructTy();

  return 0x0;
}
