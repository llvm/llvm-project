// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,both %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,both %s
#include <ptrcheck.h>

int side_effect(void);
int get_count(void);
char* get_single_ptr();
struct eb_with_other_data {
  char* __ended_by(end) start;
  char* end;
  int other;
};
void consume_eb_with_other_data(struct eb_with_other_data);

struct NestedEB {
  struct eb_with_other_data inner;
  char* end;
  char* __ended_by(end) start;
  int other;
};

struct no_attr_with_other_data {
  char* end;
  char* start;
  int other;
};
_Static_assert(sizeof(struct eb_with_other_data) == sizeof(struct no_attr_with_other_data), "size mismatch");

union TransparentUnion {
  struct eb_with_other_data cb;
  struct no_attr_with_other_data no_cb;
} __attribute__((__transparent_union__));

void receive_transparent_union(union TransparentUnion);


void no_diag(void) {
  // No diagnostics
  consume_eb_with_other_data((struct eb_with_other_data){
    0x0,
    0x0,
    0x0
  });
}

struct eb_with_other_data global_no_diags = (struct eb_with_other_data) {
  // FIXME: This should be allowed. rdar://81135826
  0x0, // both-error{{initializer element is not a compile-time constant}}
  0x0,
  0x0
};

struct eb_with_other_data field_initializers_with_side_effects(struct eb_with_other_data* s) {
   *s = (struct eb_with_other_data){
    0x0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()
  };

  struct eb_with_other_data s2 = (struct eb_with_other_data){
    0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()
  };

  consume_eb_with_other_data((struct eb_with_other_data){
    0x0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()}
  );
  consume_eb_with_other_data((struct eb_with_other_data){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect(),
    .start = 0x0,
    .end = 0x0}
  );

  (void) (struct eb_with_other_data){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect(),
    .start = 0x0,
    .end = 0x0};

  // no diags for structs without attributes
  struct no_attrs {
    char* start;
    char* end;
    int other;
  };
  (void) (struct no_attrs) { 0x0, 0x0, side_effect()};

  // Nested
  struct Contains_eb_with_other_data {
    struct eb_with_other_data s;
    int other;
  };
  (void)(struct Contains_eb_with_other_data) {
    .s = {
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .start = 0x0,
      .end = 0x0
    },
    .other = 0x0
  };

  // Nested CompoundLiteralExpr
  (void)(struct NestedEB) {
    // expected-warning@+1{{initializer '(struct eb_with_other_data){.start = 0, .end = 0, .other = side_effect()}' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .inner = (struct eb_with_other_data) {
      .start = 0x0,
      .end = 0x0,
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminat}}
      .other = side_effect()
    },
    .start = 0x0,
    .end = 0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect()
  };

  // Test array initializer list that initializes structs
  (void)(struct eb_with_other_data[]){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    {0x0, 0x0, side_effect()},
    {0x0, 0x0, 0x0}
  };

  union UnionWith_eb_with_other_data {
    struct eb_with_other_data u;
    int other;
  };

  (void)(union UnionWith_eb_with_other_data) {
    {
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .start = 0x0,
      .end = 0x0
    }
  };

  // Call a function that takes a transparent union
  receive_transparent_union(
    // Test very "untransparent"
    (union TransparentUnion) {.cb = 
      {
        // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
        .other = side_effect(),
        .start = 0x0,
        .end = 0x0
      }
    }
  );
  // Call using
  receive_transparent_union(
    // Transparent
    (struct eb_with_other_data){
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .start = 0x0,
      .end = 0x0
    }
  );

  // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
  return (struct eb_with_other_data) { 0x0, 0x0, side_effect()};
};

struct eb_with_other_data global_eb_with_other_data_side_effect_init = 
  (struct eb_with_other_data) {
    // FIXME: Location of first non-constant is wrong (rdar://81135826)
    .start = 0x0, // both-error{{initializer element is not a compile-time constant}}
    .end = 0x0,
    .other = side_effect() 
};

struct eb_with_other_data side_effects_in_ptrs(struct eb_with_other_data* s) {
    *s = (struct eb_with_other_data){
    get_single_ptr(), // expected-error{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
    get_single_ptr(), // expected-error{{initalizer for end pointer with side effects is not yet supported}}
    0x0
  };

  consume_eb_with_other_data(
    (struct eb_with_other_data){
      get_single_ptr(), // expected-error{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
      get_single_ptr(), // expected-error{{initalizer for end pointer with side effects is not yet supported}}
      0x0
    }
  );

  (void)(struct eb_with_other_data){
    get_single_ptr(), // expected-error{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
    get_single_ptr(), // expected-error{{initalizer for end pointer with side effects is not yet supported}}
    0x0
  };

  (void)(struct NestedEB) {
    .inner = (struct eb_with_other_data) {
      // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
      .start = get_single_ptr(),
      // expected-error@+1{{initalizer for end pointer with side effects is not yet supported}}
      .end = get_single_ptr(),
      .other = 0
    },
    // expected-error@+1{{initalizer for end pointer with side effects is not yet supported}}
    .end = get_single_ptr(),
    // expected-error@+1{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
    .start = get_single_ptr(),
    .other = 0
  };

  return (struct eb_with_other_data){
    get_single_ptr(), // expected-error{{initalizer for '__ended_by' pointer with side effects is not yet supported}}
    get_single_ptr(), // expected-error{{initalizer for end pointer with side effects is not yet supported}}
    0x0
  };
}

// To keep this test case small just test the `*s = CompoundLiteralExpr` variant
// in the remaining test cases.

void partial_init_start(struct eb_with_other_data* s, char* start, struct NestedEB* neb) {
  // expected-warning@+1{{implicitly initializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  *s = (struct eb_with_other_data){ .start = start };

  *neb = (struct NestedEB) {
    .inner = (struct eb_with_other_data) {
      .start = start
    },// expected-warning{{implicitly initializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
    .start = start
  }; // expected-warning{{implicitly initializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
}


// Partial init diagnostic isn't emitted because `global_start_ptr` isn't constant
char* global_start_ptr;
struct eb_with_other_data global_eb_partial_init = 
  (struct eb_with_other_data) {
    .other = 0x0,
    .start = global_start_ptr // both-error{{initializer element is not a compile-time constant}}
};
struct eb_with_other_data global_eb_partial_init2 = 
  (struct eb_with_other_data) {
    .other = 0x0,
    // FIXME: This should be treated as a constant (rdar://81135826)
    .start = __unsafe_forge_bidi_indexable(char*, 0x1, 4) // both-error{{initializer element is not a compile-time constant}}
};


void end_is_null(struct eb_with_other_data* s, char* start, struct NestedEB* neb) {
  // expected-warning@+1{{initializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
  *s = (struct eb_with_other_data){ .start = start, .end = 0x0 };

  *neb = (struct NestedEB) {
    .inner = (struct eb_with_other_data) {
      .start = start,
      // expected-warning@+1{{nitializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
      .end = 0x0
    },
    .start = start,
    // expected-warning@+1{{nitializing field 'end' of type 'char *__single /* __started_by(start) */ ' (aka 'char *__single') to NULL while 'start' is initialized with a value rarely succeeds}}
    .end = 0x0
  }; 
}

void partial_init_ended(struct eb_with_other_data* s, char* end, struct NestedEB* neb) {
  // expected-warning@+1{{implicitly initializing field 'start' of type 'char *__single __ended_by(end)' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  *s = (struct eb_with_other_data){ .end = end };


  *neb = (struct NestedEB) {
    .inner = (struct eb_with_other_data) {
      .end = end
    }, // expected-warning{{implicitly initializing field 'start' of type 'char *__single __ended_by(end)' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
    .end = end
  }; // expected-warning{{implicitly initializing field 'start' of type 'char *__single __ended_by(end)' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
}

void start_is_null(struct eb_with_other_data* s, char* end) {
  // expected-warning@+1{{initializing field 'start' of type 'char *__single __ended_by(end)' (aka 'char *__single') to NULL while 'end' is initialized with a value rarely succeeds}}
  *s = (struct eb_with_other_data){ .start = 0x0, .end = end};
}


void partial_init_other(struct eb_with_other_data* s, struct NestedEB* neb) {
  // OK
  *s = (struct eb_with_other_data){ .other = 0x0 };
  *neb = (struct NestedEB) {
    .inner = (struct eb_with_other_data) {.other = 0x0},
    .other = 0x0
  };
}

void implicit_init_all(struct eb_with_other_data* s, struct NestedEB* neb) {
  // OK
  *s = (struct eb_with_other_data){0};
  *neb = (struct NestedEB) {0};
  *neb = (struct NestedEB) {
    .inner = (struct eb_with_other_data) {0}
  };
}

void var_init_from_compound_literal_with_side_effect(char*__bidi_indexable ptr) {
  // FIXME: This diagnostic is misleading. The actually restriction is the
  // compound literal can't contain side-effects but the diagnostic talks about
  // "non-constant array". If the side-effect (call to `get_count()`) is removed
  // then this error goes away even though the compound literal is still
  // non-constant due to initializing from `ptr`.
  // both-error@+1{{cannot initialize array of type 'struct eb_with_other_data[]' with non-constant array of type 'struct eb_with_other_data[2]'}}
  struct eb_with_other_data arr[] = (struct eb_with_other_data[]){
    {.start = ptr, .end = ptr + 1, .other = 0x0},
    // expected-warning@+1{{initializer 'get_count()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    {.start = ptr, .end = ptr + 1, .other = get_count()},
  };
}
