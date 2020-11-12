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
  int buffer[N];
  int *start, *finish;

public:
  BadIterable() : start(buffer), finish(buffer + N) {}

  int* begin() { return start; }
  int* end() { return finish + 1; } // expected-warning{{Returned pointer value points outside the original object (potential buffer overflow)}}
};

void use_bad_iterable_object() {
  BadIterable<20> iter;
  iter.end();
}
