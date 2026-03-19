// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

#define __counted_by_or_null(f)  __attribute__((counted_by_or_null(f)))

// =============================================================================
// # Struct incomplete type with attribute in the decl position
// =============================================================================

// Note: This could be considered misleading. The typedef isn't actually on this
// line. Also note the discrepancy in diagnostic count (27 vs 51) is due to
// the pointer arithmetic on incomplete pointee type diagnostic always using
// diagnostic text that refers to the underlying forward decl, even when the
// typedef is used.
// expected-note@+3 27{{consider providing a complete definition for 'Incomplete_t' (aka 'struct IncompleteTy')}}
// The 'forward declaration' notes come from 'arithmetic on a pointer to an incomplete type' errors
// expected-note@+1 24{{forward declaration of 'struct IncompleteTy'}}
struct IncompleteTy; // expected-note 27{{consider providing a complete definition for 'struct IncompleteTy'}}

typedef struct IncompleteTy Incomplete_t; 

struct CBBufDeclPos {
  int count;
  struct IncompleteTy* buf __counted_by_or_null(count); // OK expected-note 27{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  Incomplete_t* buf_typedef __counted_by_or_null(count); // OK expected-note 27{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void consume_struct_IncompleteTy(struct IncompleteTy* buf);

int idx(void);



void test_CBBufDeclPos(struct CBBufDeclPos* ptr) {
  // ===========================================================================
  // ## Local variable initialization
  // ===========================================================================
  struct CBBufDeclPos explicit_desig_init = {
    .count = 0,
    // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
    .buf = 0x0,
    // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
    .buf_typedef = 0x0
  };
  // Variable is not currently marked as invalid so uses of the variable allows
  // diagnostics to fire.
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  explicit_desig_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  explicit_desig_init.buf_typedef = 0x0;
  // expected-error@+1{{cannot use 'explicit_desig_init.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  void *tmp = explicit_desig_init.buf;
  // expected-error@+1{{cannot use 'explicit_desig_init.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  void *tmp2 = explicit_desig_init.buf_typedef;

  struct CBBufDeclPos partial_explicit_desig_init = {
    .count = 0,
    // .buf and .buf_typedef are implicit zero initialized
    // expected-error@+2{{cannot implicitly initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
    // expected-error@+1{{cannot implicitly initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  };

  struct CBBufDeclPos implicit_full_init = {
    0
    // expected-error@+2{{cannot implicitly initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
    // expected-error@+1{{cannot implicitly initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  };
  // Variable is not currently marked as invalid so uses of the variable allows
  // diagnostics to fire.
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  implicit_full_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  implicit_full_init.buf_typedef = 0x0;
  // expected-error@+1{{cannot use 'implicit_full_init.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  void* tmp3 = implicit_full_init.buf;
  // expected-error@+1{{cannot use 'implicit_full_init.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  void* tmp4 = implicit_full_init.buf_typedef;
  
  struct CBBufDeclPos explicit_non_desig_init = {
    0,
    // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
    0x0,
    // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
    0x0
  };




  // ===========================================================================
  // ## Assignment to fields
  // ===========================================================================
  struct CBBufDeclPos uninit;
  uninit.count = 0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  uninit.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  uninit.buf_typedef = 0x0;
  ptr->count = 0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  ptr->buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  ptr->buf_typedef = 0x0;


  // ===========================================================================
  // ## Make sure modifying the fields through other assignment operators is not
  //    allowed
  // ===========================================================================
  uninit.buf++; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  ++uninit.buf; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  uninit.buf += 1; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  uninit.buf_typedef++; // // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  ++uninit.buf_typedef; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  uninit.buf_typedef -= 1; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  
  uninit.buf--; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  --uninit.buf; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  uninit.buf -= 1; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  uninit.buf_typedef--; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  --uninit.buf_typedef; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  uninit.buf_typedef -= 1; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}

  ptr->buf++; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  ++ptr->buf; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  ptr->buf += 1; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  ptr->buf--; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  --ptr->buf; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}
  ptr->buf -= 1; // expected-error{{arithmetic on a pointer to an incomplete type 'struct IncompleteTy'}}

  ptr->buf_typedef++; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  ++ptr->buf_typedef; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  ptr->buf_typedef += 1; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  ptr->buf_typedef--; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  --ptr->buf_typedef; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}
  ptr->buf_typedef -= 1; // expected-error{{arithmetic on a pointer to an incomplete type 'Incomplete_t' (aka 'struct IncompleteTy')}}

  // ===========================================================================
  // ## Use of fields in expressions
  // ===========================================================================
  // expected-error@+2{{cannot use 'uninit.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  void* addr = 
    ((char*) uninit.buf ) + 1;
  // expected-error@+2{{cannot use 'uninit.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  void* addr_typedef = 
    ((char*) uninit.buf_typedef ) + 1;
  // expected-error@+2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  void* addr_ptr = 
    ((char*) ptr->buf ) + 1;
  // expected-error@+2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  void* addr_ptr_typedef = 
    ((char*) ptr->buf_typedef ) + 1;


  // ===========================================================================
  // ## Take address of fields
  // ===========================================================================
  // TODO: This should be forbidden, not because of the incomplete pointee type
  // but because in the -fbounds-safety language model the address of a
  // `counted_by` pointer cannot be taken to avoid it being possible to modify
  // the `counted_by` pointer through another pointer. Whether or not this
  // should be forbidden when `-fbounds-safety` is off is TBD.
  //
  // The incomplete pointee type isn't actually a problem here for
  // `-fbounds-safety` because taking the address of a pointer returns a pointer
  // that have the bounds of a single `void*`, so bounds checks on the resulting
  // pointer don't need to know `sizeof(struct IncompleteTy)` but instead
  // `sizeof(struct IncompleteTy* buf __counted_by_or_null(count))` which is just the
  // size of a pointer.
  void* take_addr = &uninit.buf;
  void* take_addr_typedef = &uninit.buf_typedef;
  void* take_addr_ptr = &ptr->buf;
  void* take_addr_ptr_typedef = &ptr->buf_typedef;

  // expected-error@+1{{cannot use 'uninit.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  struct IncompleteTy* addr_elt_zero = &uninit.buf[0];
  // expected-error@+1{{cannot use 'uninit.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  struct IncompleteTy* addr_elt_idx = &uninit.buf[idx()];

  // expected-error@+1{{cannot use 'uninit.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  struct IncompleteTy* addr_elt_zero_typedef = &uninit.buf_typedef[0];
  // expected-error@+1{{cannot use 'uninit.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  struct IncompleteTy* addr_elt_idx_typedef = &uninit.buf_typedef[idx()];

  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  struct IncompleteTy* addr_elt_zero_ptr = &ptr->buf[0];
  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  struct IncompleteTy* addr_elt_idx_ptr = &ptr->buf[idx()];
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  struct IncompleteTy* addr_elt_zero_ptr_typedef = &ptr->buf_typedef[0];
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  struct IncompleteTy* addr_elt_idx_ptr_typedef = &ptr->buf_typedef[idx()];


  // ===========================================================================
  // ## Use fields as call arguments
  // ===========================================================================
  // expected-error@+1{{cannot use 'uninit.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  consume_struct_IncompleteTy(uninit.buf);
  // expected-error@+1{{cannot use 'uninit.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  consume_struct_IncompleteTy(uninit.buf_typedef);
  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  consume_struct_IncompleteTy(ptr->buf);
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  consume_struct_IncompleteTy(ptr->buf_typedef);

  // ===========================================================================
  // ## Use [] operator on fields
  // ===========================================================================
  // expected-error@+1 2{{cannot use 'uninit.buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  uninit.buf[0] = uninit.buf[1];
  // expected-error@+1 2{{cannot use 'uninit.buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  uninit.buf_typedef[0] = uninit.buf_typedef[1];
  // expected-error@+1 2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  ptr->buf[0] = ptr->buf[1];
  // expected-error@+1 2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  ptr->buf_typedef[0] = ptr->buf_typedef[1];
}


// =============================================================================
// ## Global initialization
// =============================================================================

struct CBBufDeclPos global_explicit_desig_init = {
  .count = 0,
  // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  .buf = 0x0,
  // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  .buf_typedef = 0x0
};

void use_global_explicit_desig_init(void) {
  // Variable isn't marked as invalid so diagnostics still fire
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  global_explicit_desig_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  global_explicit_desig_init.buf_typedef = 0x0;
}

struct CBBufDeclPos global_partial_explicit_desig_init = {
  .count = 0,
  // .buf and .buf_typedef are implicit zero initialized
  // expected-error@+2{{cannot implicitly initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  // expected-error@+1{{cannot implicitly initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
};

struct CBBufDeclPos global_implicit_full_init = {
  0
  // expected-error@+2{{cannot implicitly initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  // expected-error@+1{{cannot implicitly initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
};

struct CBBufDeclPos global_explicit_non_desig_init = {
  0,
  // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'struct IncompleteTy' is incomplete}}
  0x0,
  // expected-error@+1{{cannot initialize 'CBBufDeclPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_t * __counted_by_or_null(count)' (aka 'struct IncompleteTy *') because the pointee type 'Incomplete_t' (aka 'struct IncompleteTy') is incomplete}}
  0x0
};

extern struct CBBufDeclPos global_declaration; // OK

// TODO: These tentative definitions are implicitly empty initialized to zero.
// This should generate an error diagnostic but currently doesn't. There should
// be a carve out to allow `__counted_by_or_null(0)` which is the only constant count
// version of the attribute where it is valid to assign NULL.
struct CBBufDeclPos global_tentative_defn;
static struct CBBufDeclPos global_tentative_defn2;

// =============================================================================
// ## Completing the definition of the type allows use of CBBufDeclPos fields
// =============================================================================
struct IncompleteTy {
  int field;
};

void test_CBBufDeclPos_completed(struct CBBufDeclPos* ptr) {
  // Initialization is ok
  struct CBBufDeclPos explicit_desig_init = {
    .count = 0,
    .buf = 0x0,
    .buf_typedef = 0x0
  };

  struct CBBufDeclPos partial_explicit_desig_init = {
    .count = 0,
    // .buf and .buf_typedef are implicit zero initialized
  };

  struct CBBufDeclPos implicit_full_init = {0};
  
  struct CBBufDeclPos explicit_non_desig_init = {
    0,
    0x0,
    0x0
  };

  // Assignment to fields is ok
  ptr->buf = 0x0;
  ptr->buf_typedef = 0x0;

  // Use of fields in expressions is ok
  void* tmp = ptr->buf;
  void* tmp2 = ptr->buf_typedef;

  // Take address of fields is ok
  void* take_addr_ptr = &ptr->buf;
  void* take_addr_ptr_typedef = &ptr->buf_typedef;

  struct IncompleteTy* addr_elt_zero_ptr = &ptr->buf[0];
  struct IncompleteTy* addr_elt_idx_ptr = &ptr->buf[idx()];
  struct IncompleteTy* addr_elt_zero_ptr_typedef = &ptr->buf_typedef[0];
  struct IncompleteTy* addr_elt_idx_ptr_typedef = &ptr->buf_typedef[idx()];

  // As call arguments is ok
  consume_struct_IncompleteTy(ptr->buf);
  consume_struct_IncompleteTy(ptr->buf_typedef);

  // In [] operator is ok
  ptr->buf[0] = ptr->buf[1];
  ptr->buf_typedef[0] = ptr->buf_typedef[1];
}

// Global initialization is ok

struct CBBufDeclPos global_explicit_desig_init_completed = {
  .count = 0,
  .buf = 0x0,
  .buf_typedef = 0x0
};

struct CBBufDeclPos global_partial_explicit_desig_init_completed = {
  .count = 0,
  // .buf and .buf_typedef are implicit zero initialized
};

struct CBBufDeclPos global_implicit_full_init_completed = {0};

struct CBBufDeclPos global_explicit_non_desig_init_completed = {
  0,
  0x0,
  0x0
};

extern struct CBBufDeclPos global_declaration;
struct CBBufDeclPos global_tentative_defn;
static struct CBBufDeclPos global_tentative_defn2;

// =============================================================================
// # Struct incomplete type with attribute in the pointer position
// =============================================================================

// expected-note@+1 8{{consider providing a complete definition for 'Incomplete_ty2' (aka 'struct IncompleteTy2')}}
struct IncompleteTy2; // expected-note 8{{consider providing a complete definition for 'struct IncompleteTy2'}}
typedef struct IncompleteTy2 Incomplete_ty2;

void consume_struct_IncompleteTy2(struct IncompleteTy2* buf);

struct CBBufTyPos {
  int count;
  struct IncompleteTy2* __counted_by_or_null(count) buf ; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  Incomplete_ty2 *__counted_by_or_null(count) buf_typedef; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void use_CBBufTyPos(struct CBBufTyPos* ptr) {
  struct CBBufTyPos explicit_desig_init = {
    .count = 0,
    // expected-error@+1{{cannot initialize 'CBBufTyPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
    .buf = 0x0,
    // expected-error@+1{{cannot initialize 'CBBufTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
    .buf_typedef = 0x0
  };

  // Assignment
  // expected-error@+1{{cannot assign to 'CBBufTyPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  explicit_desig_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  explicit_desig_init.buf_typedef = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufTyPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  ptr->buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  ptr->buf_typedef = 0x0;

  // Use
  // expected-error@+2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  void* addr = 
    ((char*) ptr->buf ) + 1;
  // expected-error@+2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  void* addr_typedef = 
    ((char*) ptr->buf_typedef ) + 1;

  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  consume_struct_IncompleteTy2(ptr->buf);
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  consume_struct_IncompleteTy2(ptr->buf_typedef);

  // expected-error@+1 2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  ptr->buf[0] = ptr->buf[1];
  // expected-error@+1 2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  ptr->buf_typedef[0] = ptr->buf_typedef[1];
}

struct CBBufTyPos global_explicit_desig_init_struct_type_pos = {
  .count = 0,
  // expected-error@+1{{cannot initialize 'CBBufTyPos::buf' with '__counted_by_or_null' attributed type 'struct IncompleteTy2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'struct IncompleteTy2' is incomplete}}
  .buf = 0x0,
  // expected-error@+1{{cannot initialize 'CBBufTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'Incomplete_ty2 * __counted_by_or_null(count)' (aka 'struct IncompleteTy2 *') because the pointee type 'Incomplete_ty2' (aka 'struct IncompleteTy2') is incomplete}}
  .buf_typedef = 0x0
};

// Defining the type makes `CBBufTyPos` fields usable
struct IncompleteTy2 {
  int field;
};

void use_CBBufTyPos_completed(struct CBBufTyPos* ptr) {
  ptr->buf = 0x0;
  ptr->buf_typedef = 0x0;
  void* addr = ((char*) ptr->buf) + 1;
  void* addr_typedef = ((char*) ptr->buf_typedef) + 1;
}

// =============================================================================
// # union incomplete type
// =============================================================================

// expected-note@+1 8{{consider providing a complete definition for 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy')}}
union IncompleteUnionTy; // expected-note 8{{consider providing a complete definition for 'union IncompleteUnionTy'}}
typedef union IncompleteUnionTy IncompleteUnion_ty;

void consume_struct_IncompleteUnionTy(union IncompleteUnionTy* buf);

struct CBBufUnionTyPos {
  int count;
  union IncompleteUnionTy* __counted_by_or_null(count) buf ; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  IncompleteUnion_ty *__counted_by_or_null(count) buf_typedef; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void use_CBBufUnionTyPos(struct CBBufUnionTyPos* ptr) {
  struct CBBufUnionTyPos explicit_desig_init = {
    .count = 0,
    // expected-error@+1{{cannot initialize 'CBBufUnionTyPos::buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
    .buf = 0x0,
    // expected-error@+1{{cannot initialize 'CBBufUnionTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
    .buf_typedef = 0x0
  };

  // Assignment
  // expected-error@+1{{cannot assign to 'CBBufUnionTyPos::buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  explicit_desig_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufUnionTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  explicit_desig_init.buf_typedef = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufUnionTyPos::buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  ptr->buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufUnionTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  ptr->buf_typedef = 0x0;

  // Use
  // expected-error@+2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  void* addr = 
    ((char*) ptr->buf ) + 1;
  // expected-error@+2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  void* addr_typedef = 
    ((char*) ptr->buf_typedef ) + 1;

  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  consume_struct_IncompleteUnionTy(ptr->buf);
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  consume_struct_IncompleteUnionTy(ptr->buf_typedef);

  // expected-error@+1 2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  ptr->buf[0] = ptr->buf[1];
  // expected-error@+1 2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  ptr->buf_typedef[0] = ptr->buf_typedef[1];
}

struct CBBufUnionTyPos global_explicit_desig_init_union_type_pos = {
  .count = 0,
  // expected-error@+1{{cannot initialize 'CBBufUnionTyPos::buf' with '__counted_by_or_null' attributed type 'union IncompleteUnionTy * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'union IncompleteUnionTy' is incomplete}}
  .buf = 0x0,
  // expected-error@+1{{cannot initialize 'CBBufUnionTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteUnion_ty * __counted_by_or_null(count)' (aka 'union IncompleteUnionTy *') because the pointee type 'IncompleteUnion_ty' (aka 'union IncompleteUnionTy') is incomplete}}
  .buf_typedef = 0x0
};

// Defining the type makes `CBBufUnionTyPos` fields usable
union IncompleteUnionTy {
  int field;
};

void use_CBBufUnionTyPos_completed(struct CBBufUnionTyPos* ptr) {
  ptr->buf = 0x0;
  ptr->buf_typedef = 0x0;
  void* addr = ((char*) ptr->buf) + 1;
  void* addr_typedef = ((char*) ptr->buf_typedef) + 1;
}

// =============================================================================
// # enum incomplete type
// =============================================================================

// expected-note@+1 8{{consider providing a complete definition for 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy')}}
enum IncompleteEnumTy; // expected-note 8{{consider providing a complete definition for 'enum IncompleteEnumTy'}}
typedef enum IncompleteEnumTy IncompleteEnum_ty;

void consume_struct_IncompleteEnumTy(enum IncompleteEnumTy* buf);

struct CBBufEnumTyPos {
  int count;
  enum IncompleteEnumTy* __counted_by_or_null(count) buf ; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
  IncompleteEnum_ty *__counted_by_or_null(count) buf_typedef; // OK expected-note 8{{consider using '__sized_by_or_null' instead of '__counted_by_or_null'}}
};

void use_CBBufEnumTyPos(struct CBBufEnumTyPos* ptr) {
  struct CBBufEnumTyPos explicit_desig_init = {
    .count = 0,
    // expected-error@+1{{cannot initialize 'CBBufEnumTyPos::buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
    .buf = 0x0,
    // expected-error@+1{{cannot initialize 'CBBufEnumTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
    .buf_typedef = 0x0
  };

  // Assignment
  // expected-error@+1{{cannot assign to 'CBBufEnumTyPos::buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  explicit_desig_init.buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufEnumTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  explicit_desig_init.buf_typedef = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufEnumTyPos::buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  ptr->buf = 0x0;
  // expected-error@+1{{cannot assign to 'CBBufEnumTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  ptr->buf_typedef = 0x0;

  // Use
  // expected-error@+2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  void* addr = 
    ((char*) ptr->buf ) + 1;
  // expected-error@+2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  void* addr_typedef = 
    ((char*) ptr->buf_typedef ) + 1;

  // expected-error@+1{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  consume_struct_IncompleteEnumTy(ptr->buf);
  // expected-error@+1{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  consume_struct_IncompleteEnumTy(ptr->buf_typedef);

  // expected-error@+1 2{{cannot use 'ptr->buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  ptr->buf[0] = ptr->buf[1];
  // expected-error@+1 2{{cannot use 'ptr->buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  ptr->buf_typedef[0] = ptr->buf_typedef[1];
}

struct CBBufEnumTyPos global_explicit_desig_init_enum_type_pos = {
  .count = 0,
  // expected-error@+1{{cannot initialize 'CBBufEnumTyPos::buf' with '__counted_by_or_null' attributed type 'enum IncompleteEnumTy * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'enum IncompleteEnumTy' is incomplete}}
  .buf = 0x0,
  // expected-error@+1{{cannot initialize 'CBBufEnumTyPos::buf_typedef' with '__counted_by_or_null' attributed type 'IncompleteEnum_ty * __counted_by_or_null(count)' (aka 'enum IncompleteEnumTy *') because the pointee type 'IncompleteEnum_ty' (aka 'enum IncompleteEnumTy') is incomplete}}
  .buf_typedef = 0x0
};

// Defining the type makes `CBBufEnumTyPos` fields usable
enum IncompleteEnumTy {
  ONE,
  TWO
};

void use_CBBufEnumTyPos_completed(struct CBBufEnumTyPos* ptr) {
  ptr->buf = 0x0;
  ptr->buf_typedef = 0x0;
  void* addr = ((char*) ptr->buf) + 1;
  void* addr_typedef = ((char*) ptr->buf_typedef) + 1;
}

// Make a complete enum by providing an underlying type
enum CompleteEnumTy : unsigned;
typedef enum CompleteEnumTy CompleteEnum_ty;
struct CBBufEnumTyPos2 {
  int count;
  enum CompleteEnumTy* __counted_by_or_null(count) buf;
  CompleteEnum_ty *__counted_by_or_null(count) buf_typedef;
};

void use_CBBufEnumTyPos2(struct CBBufEnumTyPos2* ptr) {
  struct CBBufEnumTyPos2 explicit_desig_init = {
    .count = 0,
    .buf = 0x0, // OK
    .buf_typedef = 0x0 // OK
  };
}

// Make a complete enum by providing a concrete declaration
enum CompleteEnumTy2 {
  VALUE_ONE,
  VALUE_TWO
};
typedef enum CompleteEnumTy2 CompleteEnum_ty2;
struct CBBufEnumTyPos3 {
  int count;
  enum CompleteEnumTy2* __counted_by_or_null(count) buf;
  CompleteEnum_ty2 *__counted_by_or_null(count) buf_typedef;
};

void use_CBBufEnumTyPos3(struct CBBufEnumTyPos3* ptr) {
  struct CBBufEnumTyPos3 explicit_desig_init = {
    .count = 0,
    .buf = 0x0, // OK
    .buf_typedef = 0x0 // OK
  };
}


// =============================================================================
// # Array of __counted_by_or_null pointers
// =============================================================================

struct IncompleteTy3;

struct CBBufFAMofCountedByPtrs {
  int size;
  // TODO: This is misleading. The attribute is written in the type position
  // but clang currently doesn't treat it like that and it gets treated as
  // an attribute on the array, rather than on the element type.
  // expected-error@+1{{'counted_by_or_null' only applies to pointers; did you mean to use 'counted_by'?}}
  struct IncompleteTy3* __counted_by_or_null(size) arr[];
};

void arr_of_counted_by_ptr(struct CBBufFAMofCountedByPtrs* ptr) {
  // TODO: Should be disallowed once parsing attributes in the type position
  // works.
  ptr->arr[0] = 0x0;
  void* addr = ((char*) ptr->arr[0]) + 1;
}
