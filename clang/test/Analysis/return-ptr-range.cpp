// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.security.ReturnPtrRange -analyzer-output text -verify %s

int conjure_index();

namespace test_element_index_lifetime {

int arr[10]; // expected-note{{Original object declared here}} expected-note{{Original object declared here}}
int *ptr;

int *test_global_ptr() {
  do { // expected-note{{Loop condition is false.  Exiting loop}}
    int x = conjure_index();
    ptr = arr + x; // expected-note{{Value assigned to 'ptr'}}
    if (x != 20) // expected-note{{Assuming 'x' is equal to 20}}
                 // expected-note@-1{{Taking false branch}}
      return arr; // no-warning
  } while(0);
  return ptr; // expected-warning{{Returned pointer value points outside the original object (potential buffer overflow) [alpha.security.ReturnPtrRange]}}
              // expected-note@-1{{Returned pointer value points outside the original object (potential buffer overflow)}}
              // expected-note@-2{{Original object 'arr' is an array of 10 'int' objects}}
}

int *test_local_ptr() {
  int *local_ptr;
  do { // expected-note{{Loop condition is false.  Exiting loop}}
    int x = conjure_index();
    local_ptr = arr + x; // expected-note{{Value assigned to 'local_ptr'}}
    if (x != 20) // expected-note{{Assuming 'x' is equal to 20}}
                 // expected-note@-1{{Taking false branch}}
      return arr; // no-warning
  } while(0);
  return local_ptr; // expected-warning{{Returned pointer value points outside the original object (potential buffer overflow) [alpha.security.ReturnPtrRange]}}
                    // expected-note@-1{{Returned pointer value points outside the original object (potential buffer overflow)}}
                    // expected-note@-2{{Original object 'arr' is an array of 10 'int' objects}}
}

}

template <typename T, int N>
T* end(T (&arr)[N]) {
  return arr + N; // no-warning, because we want to avoid false positives on returning the end() iterator of a container.
}

void get_end_of_array() {
  static int arr[10];
  end(arr);
}

template <int N>
class Iterable {
  int buffer[N];
  int *start, *finish;

public:
  Iterable() : start(buffer), finish(buffer + N) {}

  int* begin() { return start; }
  int* end() { return finish; }
};

void use_iterable_object() {
  Iterable<20> iter;
  iter.end();
}

template <int N>
class BadIterable {
  int buffer[N]; // expected-note{{Original object declared here}}
  int *start, *finish;

public:
  BadIterable() : start(buffer), finish(buffer + N) {} // expected-note{{Value assigned to 'iter.finish'}}

  int* begin() { return start; }
  int* end() { return finish + 1; } // expected-warning{{Returned pointer value points outside the original object}}
                                    // expected-note@-1{{Returned pointer value points outside the original object}}
                                    // expected-note@-2{{Original object 'buffer' is an array of 20 'int' objects, returned pointer points at index 21}}
};

void use_bad_iterable_object() {
  BadIterable<20> iter; // expected-note{{Calling default constructor for 'BadIterable<20>'}}
                        // expected-note@-1{{Returning from default constructor for 'BadIterable<20>'}}
  iter.end(); // expected-note{{Calling 'BadIterable::end'}}
}

int *test_idx_sym(int I) {
  static int arr[10]; // expected-note{{Original object declared here}} expected-note{{'arr' initialized here}}

  if (I != 11) // expected-note{{Assuming 'I' is equal to 11}}
               // expected-note@-1{{Taking false branch}}
    return arr;
  return arr + I; // expected-warning{{Returned pointer value points outside the original object}}
                  // expected-note@-1{{Returned pointer value points outside the original object}}
                  // expected-note@-2{{Original object 'arr' is an array of 10 'int' objects, returned pointer points at index 11}}
}

namespace test_array_of_struct {

struct Data {
  int A;
  char *B;
};

Data DataArr[10]; // expected-note{{Original object declared here}}

Data *test_struct_array() {
  int I = conjure_index();
  if (I != 11) // expected-note{{Assuming 'I' is equal to 11}}
               // expected-note@-1{{Taking false branch}}
    return DataArr;
  return DataArr + I; // expected-warning{{Returned pointer value points outside the original object}}
                      // expected-note@-1{{Returned pointer value points outside the original object}}
                      // expected-note@-2{{Original object 'DataArr' is an array of 10 'test_array_of_struct::Data' objects, returned pointer points at index 11}}
}

}

