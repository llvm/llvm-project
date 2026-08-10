// RUN: %clang_cc1 -fsyntax-only -verify %s

#define INT_TAG 42

static const int test_in
    __attribute__((type_tag_for_datatype(test, int))) = INT_TAG;

// Argument index: 1, Type tag index: 2
void test_bounds_index(...)
    __attribute__((argument_with_type_tag(test, 1, 2)));

// Argument index: 1, Type tag index: 2
void test_bounds_index_ptr(void *, ...)
    __attribute__((pointer_with_type_tag(test, 1, 2)));

// Argument index: 3, Type tag index: 1
void test_bounds_arg_index(...)
    __attribute__((argument_with_type_tag(test, 3, 1)));

class C {
public:
  // Argument index: 2, Type tag index: 3
  void test_bounds_index(...)
      __attribute__((argument_with_type_tag(test, 2, 3)));

  // Argument index: 2, Type tag index: 3
  void test_bounds_index_ptr(void *, ...)
      __attribute__((pointer_with_type_tag(test, 2, 3)));

  // Argument index: 4, Type tag index: 2
  void test_bounds_arg_index(...)
      __attribute__((argument_with_type_tag(test, 4, 2)));
};

void test_bounds()
{
  C c;

  // Test the boundary edges (ensure no off-by-one) with argument indexing.
  test_bounds_index(1, INT_TAG);
  c.test_bounds_index(1, INT_TAG);
  test_bounds_index_ptr(0, INT_TAG);
  c.test_bounds_index_ptr(0, INT_TAG);

  test_bounds_index(1);       // expected-error {{type tag index 2 is greater than the number of arguments specified}}
  c.test_bounds_index(1);     // expected-error {{type tag index 3 is greater than the number of arguments specified}}
  test_bounds_index_ptr(0);   // expected-error {{type tag index 2 is greater than the number of arguments specified}}
  c.test_bounds_index_ptr(0); // expected-error {{type tag index 3 is greater than the number of arguments specified}}

  test_bounds_arg_index(INT_TAG, 1);   // expected-error {{argument index 3 is greater than the number of arguments specified}}
  c.test_bounds_arg_index(INT_TAG, 1); // expected-error {{argument index 4 is greater than the number of arguments specified}}
}
