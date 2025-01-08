// RUN: %clang_cc1 -std=c++23 -verify %s

// _fetch_add

constexpr bool test_int_fetch_add(int arg) {
  int array[8] = {1,2,3,4,5,6,7,8};
  int * ptr = &array[0];
  int * old = __atomic_fetch_add(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 5 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}
  return true;
}

static_assert(test_int_fetch_add(5)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_int_fetch_add(5)'}}

static_assert(test_int_fetch_add(sizeof(int)));

constexpr bool test_long_long_fetch_add(int arg) {
  long long array[8] = {1,2,3,4,5,6,7,8};
  long long * ptr = &array[0];
  long long * old = __atomic_fetch_add(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 6 not aligned to size of pointee type (sizeof 'long long' is {{[0-9]+}})}}
  return true;
}

static_assert(test_long_long_fetch_add(6)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_long_long_fetch_add(6)'}}


static_assert(test_long_long_fetch_add(sizeof(long long) * 4));

// _fetch_sub

constexpr bool test_int_fetch_sub(int arg) {
  int array[8] = {1,2,3,4,5,6,7,8};
  int * ptr = &array[7];
  int * old = __atomic_fetch_sub(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 7 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}
  return true;
}

static_assert(test_int_fetch_sub(7)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_int_fetch_sub(7)'}}

static_assert(test_int_fetch_sub(sizeof(int)));

constexpr bool test_long_long_fetch_sub(int arg) {
  long long array[8] = {1,2,3,4,5,6,7,8};
  long long * ptr = &array[7];
  long long * old = __atomic_fetch_sub(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 11 not aligned to size of pointee type (sizeof 'long long' is {{[0-9]+}})}}
  return true;
}

static_assert(test_long_long_fetch_sub(11)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_long_long_fetch_sub(11)'}}


static_assert(test_long_long_fetch_sub(sizeof(long long) * 4));


// _add_fetch

constexpr bool test_int_add_fetch(int arg) {
  int array[8] = {1,2,3,4,5,6,7,8};
  int * ptr = &array[0];
  int * old = __atomic_add_fetch(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 5 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}
  return true;
}

static_assert(test_int_add_fetch(5)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_int_add_fetch(5)'}}

static_assert(test_int_add_fetch(sizeof(int)));

constexpr bool test_long_long_add_fetch(int arg) {
  long long array[8] = {1,2,3,4,5,6,7,8};
  long long * ptr = &array[0];
  long long * old = __atomic_add_fetch(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 6 not aligned to size of pointee type (sizeof 'long long' is {{[0-9]+}})}}
  return true;
}

static_assert(test_long_long_add_fetch(6)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_long_long_add_fetch(6)'}}


static_assert(test_long_long_add_fetch(sizeof(long long) * 4));

// _sub_fetch

constexpr bool test_int_sub_fetch(int arg) {
  int array[8] = {1,2,3,4,5,6,7,8};
  int * ptr = &array[7];
  int * old = __atomic_sub_fetch(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 7 not aligned to size of pointee type (sizeof 'int' is {{[0-9]+}})}}
  return true;
}

static_assert(test_int_sub_fetch(7)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_int_sub_fetch(7)'}}

static_assert(test_int_sub_fetch(sizeof(int)));

constexpr bool test_long_long_sub_fetch(int arg) {
  long long array[8] = {1,2,3,4,5,6,7,8};
  long long * ptr = &array[7];
  long long * old = __atomic_sub_fetch(&ptr, arg, __ATOMIC_RELAXED); // expected-note-re {{atomic pointer operation with argument 11 not aligned to size of pointee type (sizeof 'long long' is {{[0-9]+}})}}
  return true;
}

static_assert(test_long_long_sub_fetch(11)); // expected-error {{static assertion expression is not an integral constant expression}} expected-note {{in call to 'test_long_long_sub_fetch(11)'}}


static_assert(test_long_long_sub_fetch(sizeof(long long) * 4));
