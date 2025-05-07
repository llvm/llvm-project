
// TODO: We should get the same diagnostics with/without compound_literal_init (rdar://138982703)
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=expected,both -fbounds-safety-bringup-missing-checks=compound_literal_init %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=both %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=both %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=expected,both -fbounds-safety-bringup-missing-checks=compound_literal_init %s
#include <ptrcheck.h>

int side_effect(void);
int get_count(void);
char* get_single_ptr();
struct sbon_with_other_data {
  int count;
  char* __sized_by_or_null(count) buf;
  int other;
};
void consume_sbon_with_other_data(struct sbon_with_other_data);

struct NestedSBON {
  struct sbon_with_other_data inner;
  int count;
  char* __sized_by_or_null(count) buf;
  int other;
};

struct no_attr_with_other_data {
  int count;
  char* buf;
  int other;
};
_Static_assert(sizeof(struct sbon_with_other_data) == sizeof(struct no_attr_with_other_data), "size mismatch");

union TransparentUnion {
  struct sbon_with_other_data cb;
  struct no_attr_with_other_data no_cb;
} __attribute__((__transparent_union__));

void receive_transparent_union(union TransparentUnion);


void no_diag(void) {
  // No diagnostics
  consume_sbon_with_other_data((struct sbon_with_other_data){
    0x0,
    0x0,
    0x0
  });
}

struct sbon_with_other_data global_no_diags = (struct sbon_with_other_data) {
  0x0,
  0x0,
  0x0
};

struct sbon_with_other_data field_initializers_with_side_effects(struct sbon_with_other_data* s) {
  *s = (struct sbon_with_other_data){
    0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()
  };

  struct sbon_with_other_data s2 = (struct sbon_with_other_data){
    0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()
  };

  consume_sbon_with_other_data((struct sbon_with_other_data){
    0,
    0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    side_effect()}
  );
  consume_sbon_with_other_data((struct sbon_with_other_data){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect(),
    .buf = 0x0,
    .count = 0}
  );

  (void) (struct sbon_with_other_data){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect(),
    .buf = 0x0,
    .count = 0};

  // no diags for structs without attributes
  struct no_attrs {
    int count;
    char* buf;
  };
  (void) (struct no_attrs) { side_effect(), 0x0};

  // Nested
  struct Contains_sbon_with_other_data {
    struct sbon_with_other_data s;
    int other;
  };
  (void)(struct Contains_sbon_with_other_data) {
    .s = {
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .buf = 0x0,
      .count = 0
    },
    .other = 0x0
  };

  // Nested CompoundLiteralExpr
  (void)(struct NestedSBON) {
    // expected-warning@+1{{initializer '(struct sbon_with_other_data){.count = 0, .buf = 0, .other = side_effect()}' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .inner = (struct sbon_with_other_data) {
      .count = 0,
      .buf = 0x0,
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminat}}
      .other = side_effect()
    },
    .count = 0,
    .buf = 0x0,
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    .other = side_effect()
  };

  // Test array initializer list that initializes structs
  (void)(struct sbon_with_other_data[]){
    // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    {0, 0x0, side_effect()},
    {0, 0x0, 0x0}
  };

  union UnionWith_sbon_with_other_data {
    struct sbon_with_other_data u;
    int other;
  };

  (void)(union UnionWith_sbon_with_other_data) {
    {
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .buf = 0x0,
      .count = 0
    }
  };

  // Call a function that takes a transparent union
  receive_transparent_union(
    // Test very "untransparent"
    (union TransparentUnion) {.cb = 
      {
        // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
        .other = side_effect(),
        .buf = 0x0,
        .count = 0
      }
    }
  );
  // Call using
  receive_transparent_union(
    // Transparent
    (struct sbon_with_other_data){
      // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
      .other = side_effect(),
      .buf = 0x0,
      .count = 0
    }
  );

  // expected-warning@+1{{initializer 'side_effect()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
  return (struct sbon_with_other_data) { 0, 0x0, side_effect()};
}

struct sbon_with_other_data global_sbon_with_other_data_side_effect_init = 
  (struct sbon_with_other_data) {
    .buf = 0x0,
    .count = 0,
    .other = side_effect() // both-error{{initializer element is not a compile-time constant}}
};


struct sbon_with_other_data side_effects_in_ptr(struct sbon_with_other_data* s) {
  *s = (struct sbon_with_other_data){
    0,
    // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    get_single_ptr(),
    0x0
  };

  consume_sbon_with_other_data((struct sbon_with_other_data){
    0,
    // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    get_single_ptr(),
    0x0}
  );

  (void) (struct sbon_with_other_data){
    0,
    // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    get_single_ptr(),
    0x0
  };

  (void)(struct NestedSBON) {
    .inner = (struct sbon_with_other_data) {
      .count = 0,
      // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
      .buf = get_single_ptr(),
      .other = 0
    },
    .count = 0,
    // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    .buf = get_single_ptr(),
    .other = 0
  };

  return (struct sbon_with_other_data ){
    0,
    // expected-error@+1{{initalizer for '__sized_by_or_null' pointer with side effects is not yet supported}}
    get_single_ptr(),
    0x0
  };
}

struct sbon_with_other_data side_effects_in_count(struct sbon_with_other_data* s) {
  *s = (struct sbon_with_other_data){
    // expected-error@+1{{initalizer for size with side effects is not yet supported}}
    get_count(),
    0x0,
    0x0
  };

  consume_sbon_with_other_data((struct sbon_with_other_data){
    // expected-error@+1{{initalizer for size with side effects is not yet supported}}
    get_count(),
    0x0,
    0x0}
  );

  (void) (struct sbon_with_other_data){
    // expected-error@+1{{initalizer for size with side effects is not yet supported}}
    get_count(),
    0x0,
    0x0
  };

  (void)(struct NestedSBON) {
    .inner = (struct sbon_with_other_data) {
      // expected-error@+1{{initalizer for size with side effects is not yet supported}}
      .count = get_count(),
      .buf = 0x0,
      .other = 0
    },
    // expected-error@+1{{initalizer for size with side effects is not yet supported}}
    .count = get_count(),
    .buf = 0x0,
    .other = 0
  };

  return (struct sbon_with_other_data ){
    // expected-error@+1{{initalizer for size with side effects is not yet supported}}
    get_count(),
    0x0,
    0x0
  };
}


// To keep this test case small just test the `*s = CompoundLiteralExpr` variant
// in the remaining test cases.

void constant_neg_count(struct sbon_with_other_data* s) {
  *s = (struct sbon_with_other_data){
    -1,
    // ok
    0x0,
    0x0
  };
}

struct sbon_with_other_data global_sbon_with_other_data_constant_neg_count = 
  (struct sbon_with_other_data) {
    .buf = 0x0, // ok
    .count = -1,
    .other = 0x0
};

struct NestedSBON global_nested_sbon_constant_neg_count = (struct NestedSBON) {
  .inner = (struct sbon_with_other_data) {
    .count = -1,
    .buf = 0x0, // OK
    .other = 0x0
  },
  .count = -1,
  .buf = 0x0, // OK
  .other = 0
};



void constant_pos_count_nullptr(struct sbon_with_other_data* s) {
  *s = (struct sbon_with_other_data){
    100,
    // ok
    0x0,
    0x0
  };
}

struct sbon_with_other_data global_sbon_with_other_data_constant_pos_count_nullptr = 
  (struct sbon_with_other_data) {
    .buf = 0x0, // ok
    .count = 100,
    .other = 0x0
};

struct NestedSBON global_nested_sbon_constant_pos_count_nullptr = (struct NestedSBON) {
  .inner = (struct sbon_with_other_data) {
    .count = 100,
    .buf = 0x0, // OK
    .other = 0x0
    },
  .count = 100,
  .buf = 0x0, // OK
  .other = 0
};


void bad_arr_size(struct sbon_with_other_data* s) {
  char arr[3]; // expected-note{{'arr' declared here}}
    *s = (struct sbon_with_other_data){
    100,
    // expected-error@+1{{initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') and size value of 100 with array 'arr' (which has 3 bytes) always fails}}
    arr,
    0x0
  };
}

// Can't refer to non constant expressions at file scope so the bad `count`
// diagnostic doesn't get emitted.
char global_arr[3] = {0};
struct sbon_with_other_data global_sbon_with_other_data_bad_arr_size = 
  (struct sbon_with_other_data) {
    .buf = global_arr, // both-error{{initializer element is not a compile-time constant}}
    .count = 100,
    .other = 0x0
};


void bad_count_and_single_ptr_src(struct sbon_with_other_data* s, char* ptr) {
  *s = (struct sbon_with_other_data){
    100,
    ptr,
    0x0
  };
}


void bad_implicit_zero_count(struct sbon_with_other_data* s, char*__bidi_indexable ptr, struct NestedSBON* nsbon) {
  *s = (struct sbon_with_other_data){
    // expected-warning@+1{{'char *__single __sized_by_or_null(count)' (aka 'char *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
    .buf = ptr
  };

    *nsbon = (struct NestedSBON) {
    // expected-warning@+1{{possibly initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
    .buf = ptr,
    .inner = (struct sbon_with_other_data) {
      // expected-warning@+1{{possibly initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
      .buf = ptr
    }
  };
}


void unknown_count_and_single_ptr_src(struct sbon_with_other_data* s, char* ptr, struct NestedSBON* nsbon) {
  int count = get_count();
  *s = (struct sbon_with_other_data){
    count, // expected-note{{size initialized here}}
    // expected-warning@+1{{size value is not statically known: initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') with 'char *__single' is invalid for any size greater than 1}}
    ptr,
    0x0
  };

    *nsbon = (struct NestedSBON) {
    .inner = (struct sbon_with_other_data) {
      // expected-warning@+1{{size value is not statically known: initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') with 'char *__single' is invalid for any size greater than 1 unless the pointer is null}}
      .buf = ptr,
      .count = count // expected-note{{size initialized here}}
    },
    // expected-warning@+1{{size value is not statically known: initializing 'char *__single __sized_by_or_null(count)' (aka 'char *__single') with 'char *__single' is invalid for any size greater than 1 unless the pointer is null}}
    .buf = ptr,
    .count = count // expected-note{{size initialized here}}
  };
}

void var_init_from_compound_literal_with_side_effect(char*__bidi_indexable ptr) {
  // FIXME: This diagnostic is misleading. The actually restriction is the
  // compound literal can't contain side-effects but the diagnostic talks about
  // "non-constant array". If the side-effect (call to `get_count()`) is removed
  // then this error goes away even though the compound literal is still
  // non-constant due to initializing from `ptr`.
  // both-error@+1{{cannot initialize array of type 'struct sbon_with_other_data[]' with non-constant array of type 'struct sbon_with_other_data[2]'}}
  struct sbon_with_other_data arr[] = (struct sbon_with_other_data[]){
    {.buf = ptr, .count = 0x0, .other = 0x0},
    // expected-warning@+1{{initializer 'get_count()' has a side effect; this may lead to an unexpected result because the evaluation order of initialization list expressions is indeterminate}}
    {.buf = ptr, .count = 0x0, .other = get_count()},
  };
}

void call_null_ptr_sbon(int new_count) {
  consume_sbon_with_other_data((struct sbon_with_other_data) {
    .buf = (char*)0,
    .count = new_count
  });
  consume_sbon_with_other_data((struct sbon_with_other_data) {
    .buf = (void*)0,
    .count = new_count
  });
  consume_sbon_with_other_data((struct sbon_with_other_data) {
    .buf = ((char*)(void*)(char*)0),
    .count = new_count
  });
}
