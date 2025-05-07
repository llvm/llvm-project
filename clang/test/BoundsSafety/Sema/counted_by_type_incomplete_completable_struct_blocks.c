
// RUN: %clang_cc1 -fsyntax-only -fblocks -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
//------------------------------------------------------------------------------
// Test diagnostics on _counted_by(_or_null) pointers with an incomplete struct
// pointee type on block parameters/return type
//------------------------------------------------------------------------------

struct IncompleteStructTy; // expected-note 6{{consider providing a complete definition for 'struct IncompleteStructTy'}}

//------------------------------------------------------------------------------
// Attribute on block parameters
//------------------------------------------------------------------------------

typedef void(^cb_block_fn_t)(struct IncompleteStructTy* __counted_by(size), int size); // OK
typedef void(^cbon_block_fn_t)(struct IncompleteStructTy* __counted_by_or_null(size), int size); // OK

void consume_cb(struct IncompleteStructTy* __counted_by(size), int size);
void consume_cbon(struct IncompleteStructTy* __counted_by_or_null(size), int size);

void use_block_params_cb(void) {
  // expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'buf' with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cb_block_fn_t f_named = ^(struct IncompleteStructTy* __counted_by(size) buf, int size) { // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
    // Uses don't generate diagnostics because the parameter is treated as invalid.
    buf = 0x0;
    buf[0] = 0;
    struct IncompleteStructTy* block_local = buf;
    consume_cb(buf);
  };

  // expected-error@+2{{cannot apply '__counted_by' attribute to parameter with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-warning@+1{{omitting the parameter name in a function definition is a C23 extension}}
  cb_block_fn_t f_unnamed = ^(struct IncompleteStructTy* __counted_by(size), int size) { // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
  };

  void (^f_var_no_typedef_decl)(struct IncompleteStructTy* __counted_by(size), int size) = 
    // expected-error@+1{{cannot apply '__counted_by' attribute to parameter 'buf' with type 'struct IncompleteStructTy *__single __counted_by(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
    ^(struct IncompleteStructTy* __counted_by(size) buf, int size) { // expected-note{{consider using '__sized_by' instead of '__counted_by'}}
      // Uses don't generate diagnostics because the parameter is treated as invalid.
      buf = 0x0;
      buf[0] = 0;
      struct IncompleteStructTy* block_local = buf;
      consume_cb(buf);
    };
}

void use_block_params_cbon(void) {
  // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'buf' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  cbon_block_fn_t f_named = ^(struct IncompleteStructTy* __counted_by_or_null(size) buf, int size) { // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
    // Uses don't generate diagnostics because the parameter is treated as invalid.
    buf = 0x0;
    buf[0] = 0;
    struct IncompleteStructTy* block_local = buf;
    consume_cb(buf);
  };

  // expected-error@+2{{cannot apply '__counted_by_or_null' attribute to parameter with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
  // expected-warning@+1{{omitting the parameter name in a function definition is a C23 extension}}
  cbon_block_fn_t f_unnamed = ^(struct IncompleteStructTy* __counted_by_or_null(size), int size) { // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  };

  void (^f_var_no_typedef_decl)(struct IncompleteStructTy* __counted_by_or_null(size), int size) = 
    // expected-error@+1{{cannot apply '__counted_by_or_null' attribute to parameter 'buf' with type 'struct IncompleteStructTy *__single __counted_by_or_null(size)' (aka 'struct IncompleteStructTy *__single') on a function definition because the pointee type 'struct IncompleteStructTy' is incomplete}}
    ^(struct IncompleteStructTy* __counted_by_or_null(size) buf, int size) { // expected-note{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
      // Uses don't generate diagnostics because the parameter is treated as invalid.
      buf = 0x0;
      buf[0] = 0;
      struct IncompleteStructTy* block_local = buf;
      consume_cb(buf);
    };
}

//------------------------------------------------------------------------------
// Attribute on block return type
//------------------------------------------------------------------------------

// TODO: We should probably lift this restriction. rdar://132927574
// expected-error@+1{{'__counted_by' inside typedef is only allowed for function type}}
typedef struct IncompleteStructTy* __counted_by(size)(^cb_block_ret_fn_t)(int size); 

// Don't test this because it causes clang to crash. rdar://132927229
// void try_block_ret(void) {
//   struct IncompleteStructTy*__counted_by(size) (^f_var_no_typedef_decl)(int size) = ^ struct IncompleteStructTy*__counted_by(size) (int size) {

//   };
// }
