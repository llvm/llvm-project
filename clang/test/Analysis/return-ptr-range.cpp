// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.security.ReturnPtrRange -verify %s

int arr[10];
int *ptr;

int conjure_index();

int *test_element_index_lifetime() {
  do {
    int x = conjure_index();
    ptr = arr + x;
    if (x != 20)
      return arr; // no-warning
  } while (0);
  return ptr; // expected-warning{{Returned pointer value points outside the original object (potential buffer overflow)}}
}

int *test_element_index_lifetime_with_local_ptr() {
  int *local_ptr;
  do {
    int x = conjure_index();
    local_ptr = arr + x;
    if (x != 20)
      return arr; // no-warning
  } while (0);
  return local_ptr; // expected-warning{{Returned pointer value points outside the original object (potential buffer overflow)}}
}
