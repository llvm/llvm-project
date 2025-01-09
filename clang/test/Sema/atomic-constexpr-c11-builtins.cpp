// RUN: %clang_cc1 -std=c++2c %s

// expected-no-diagnostics

constexpr int int_min = -2147483648;
constexpr int int_max = 2147483647;

const int array[2] = {1,2};
const char small_array[2] = {1,2};

template <typename T> struct identity {
  using type = T;
};

template <typename T> using do_not_deduce = typename identity<T>::type;

// -- SIGNAL and THREAD fence --
consteval int fence_test(int v) {
  // both are no-op in constexpr
  __c11_atomic_thread_fence(__ATOMIC_ACQUIRE); 
  __c11_atomic_signal_fence(__ATOMIC_ACQUIRE);
  return v;
}

static_assert(fence_test(42) == 42);

// -- LOAD --

template <typename T> consteval T init(T value) {
  auto * av = new _Atomic T;
  __c11_atomic_init(av, value);
  auto result = __c11_atomic_load(av, __ATOMIC_RELAXED);
  delete av;
  return result;
}

// integers
static_assert(init(true) == true);
static_assert(init(false) == false);

static_assert(init(42) == 42);
static_assert(init(-128) == -128);

static_assert(init(42u) == 42u);
static_assert(init(0xFFFFFFFFu) == 0xFFFFFFFFu);

// floats
static_assert(init(42.3) == 42.3);
static_assert(init(42.3f) == 42.3f);

// pointers
static_assert(init(&array[0]) == &array[0]);
static_assert(init(&small_array[1]) == &small_array[1]);

// -- LOAD --

template <typename T> consteval T load(T value) {
  _Atomic(T) av = value;
  return __c11_atomic_load(&av, __ATOMIC_RELAXED);
}

// integers
static_assert(load(true) == true);
static_assert(load(false) == false);

static_assert(load(42) == 42);
static_assert(load(-128) == -128);

static_assert(load(42u) == 42u);
static_assert(load(0xFFFFFFFFu) == 0xFFFFFFFFu);

// floats
static_assert(load(42.3) == 42.3);
static_assert(load(42.3f) == 42.3f);

// pointers
static_assert(load(&array[0]) == &array[0]);
static_assert(load(&small_array[1]) == &small_array[1]);

// -- STORE --

template <typename T> consteval T store(T value) {
  _Atomic(T) av = T{};
  __c11_atomic_store(&av, value, __ATOMIC_RELAXED);
  return __c11_atomic_load(&av, __ATOMIC_RELAXED);
}

// integers
static_assert(store(true) == true);
static_assert(store(false) == false);

static_assert(store(42) == 42);
static_assert(store(-128) == -128);

static_assert(store(42u) == 42u);
static_assert(store(0xFFFFFFFFu) == 0xFFFFFFFFu);

// floats
static_assert(store(42.3) == 42.3);
static_assert(store(42.3f) == 42.3f);

// pointers
static_assert(store(&array[0]) == &array[0]);
static_assert(store(&small_array[1]) == &small_array[1]);

// -- EXCHANGE --
template <typename T> struct two_values {
  T before;
  T after;
  constexpr friend bool operator==(two_values, two_values) = default;
};

template <typename T> consteval auto exchange(T value, do_not_deduce<T> replacement) -> two_values<T> {
  _Atomic(T) av = T{value};
  T previous = __c11_atomic_exchange(&av, replacement, __ATOMIC_RELAXED);
  return two_values<T>{previous, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

// integers
static_assert(exchange(true,false) == two_values{true, false});
static_assert(exchange(false,true) == two_values{false, true});

static_assert(exchange(10,42) == two_values{10,42});
static_assert(exchange(14,-128) == two_values{14,-128});


static_assert(exchange(56u,42u) == two_values{56u,42u});
static_assert(exchange(0xFFu, 0xFFFFFFFFu) == two_values{0xFFu,0xFFFFFFFFu});

// floats
static_assert(exchange(42.3,1.2) == two_values{42.3,1.2});
static_assert(exchange(42.3f,-16.7f) == two_values{42.3f, -16.7f});

// pointers
static_assert(exchange(&array[0], &array[1]) == two_values{&array[0],&array[1]});
static_assert(exchange(&small_array[1], &small_array[0]) == two_values{&small_array[1], &small_array[0]});

// -- COMPARE-EXCHANGE --
template <typename T> struct comp_result {
  bool success;
  T output;
  T after;
  
  constexpr comp_result(bool success_, T output_, do_not_deduce<T> after_): success{success_}, output{output_}, after{after_} { }
  
  constexpr friend bool operator==(comp_result, comp_result) = default;
};

template <typename T> constexpr auto comp_success(T output, do_not_deduce<T> after) {
  return comp_result<T>{true, output, after};
}

template <typename T> constexpr auto comp_failure(T after) {
  return comp_result<T>{false, after, after};
}

template <typename T> consteval auto compare_exchange_weak(T original, do_not_deduce<T> expected, do_not_deduce<T> replacement) -> comp_result<T> {
  _Atomic(T) av = T{original};
  const bool success = __c11_atomic_compare_exchange_weak(&av, &expected, replacement, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return comp_result<T>{success, expected, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

// integers
static_assert(compare_exchange_weak(true, true, false) == comp_success(true, false));
static_assert(compare_exchange_weak(false, false, true) == comp_success(false, true));
static_assert(compare_exchange_weak(false, true, false) == comp_failure(false));
static_assert(compare_exchange_weak(true, false, true) == comp_failure(true));

static_assert(compare_exchange_weak(10,10,42) == comp_success(10,42));
static_assert(compare_exchange_weak(14,14,-128) == comp_success(14,-128));
static_assert(compare_exchange_weak(-10,10,42) == comp_failure(-10));
static_assert(compare_exchange_weak(-14,14,-128) == comp_failure(-14));

static_assert(compare_exchange_weak(56u, 56u,42u) == comp_success(56u,42u));
static_assert(compare_exchange_weak(0xFFu, 0xFFu, 0xFFFFFFFFu) == comp_success(0xFFu,0xFFFFFFFFu));
static_assert(compare_exchange_weak(3u, 56u,42u) == comp_failure(3u));
static_assert(compare_exchange_weak(0xFu, 0xFFu, 0xFFFFFFFFu) == comp_failure(0xFu));

// floats
static_assert(compare_exchange_weak(42.3, 42.3,1.2) == comp_success(42.3,1.2));
static_assert(compare_exchange_weak(42.3f, 42.3f,-16.7f) == comp_success(42.3f, -16.7f));
static_assert(compare_exchange_weak(0.0, 42.3,1.2) == comp_failure(0.0));
static_assert(compare_exchange_weak(0.0f, 42.3f,-16.7f) == comp_failure(0.0f));

// pointers
static_assert(compare_exchange_weak(&array[0], &array[0], &array[1]) == comp_success(&array[0],&array[1]));
static_assert(compare_exchange_weak(&small_array[1], &small_array[1], &small_array[0]) == comp_success(&small_array[1], &small_array[0]));
static_assert(compare_exchange_weak(&array[1], &array[0], &array[1]) == comp_failure(&array[1]));
static_assert(compare_exchange_weak(&small_array[0], &small_array[1], &small_array[0]) == comp_failure(&small_array[0]));


template <typename T> consteval auto compare_exchange_strong(T original, do_not_deduce<T> expected, do_not_deduce<T> replacement) -> comp_result<T> {
  _Atomic(T) av = T{original};
  const bool success = __c11_atomic_compare_exchange_strong(&av, &expected, replacement, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return comp_result<T>{success, expected, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

// integers
static_assert(compare_exchange_strong(true, true, false) == comp_success(true, false));
static_assert(compare_exchange_strong(false, false, true) == comp_success(false, true));
static_assert(compare_exchange_strong(false, true, false) == comp_failure(false));
static_assert(compare_exchange_strong(true, false, true) == comp_failure(true));

static_assert(compare_exchange_strong(10,10,42) == comp_success(10,42));
static_assert(compare_exchange_strong(14,14,-128) == comp_success(14,-128));
static_assert(compare_exchange_strong(-10,10,42) == comp_failure(-10));
static_assert(compare_exchange_strong(-14,14,-128) == comp_failure(-14));

static_assert(compare_exchange_strong(56u, 56u,42u) == comp_success(56u,42u));
static_assert(compare_exchange_strong(0xFFu, 0xFFu, 0xFFFFFFFFu) == comp_success(0xFFu,0xFFFFFFFFu));
static_assert(compare_exchange_strong(3u, 56u,42u) == comp_failure(3u));
static_assert(compare_exchange_strong(0xFu, 0xFFu, 0xFFFFFFFFu) == comp_failure(0xFu));

// floats
static_assert(compare_exchange_strong(42.3, 42.3,1.2) == comp_success(42.3,1.2));
static_assert(compare_exchange_strong(42.3f, 42.3f,-16.7f) == comp_success(42.3f, -16.7f));
static_assert(compare_exchange_strong(0.0, 42.3,1.2) == comp_failure(0.0));
static_assert(compare_exchange_strong(0.0f, 42.3f,-16.7f) == comp_failure(0.0f));

// pointers
static_assert(compare_exchange_strong(&array[0], &array[0], &array[1]) == comp_success(&array[0],&array[1]));
static_assert(compare_exchange_strong(&small_array[1], &small_array[1], &small_array[0]) == comp_success(&small_array[1], &small_array[0]));
static_assert(compare_exchange_strong(&array[1], &array[0], &array[1]) == comp_failure(&array[1]));
static_assert(compare_exchange_strong(&small_array[0], &small_array[1], &small_array[0]) == comp_failure(&small_array[0]));


// --FETCH-OP--
template <typename T, typename Y> consteval auto fetch_add(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_add(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

// integers
static_assert(fetch_add(false, 1) == two_values{false, true});
static_assert(fetch_add(0, 100) == two_values{0, 100});
static_assert(fetch_add(100, -50) == two_values{100, 50});

static_assert(fetch_add(int_max, 1) == two_values{int_max, int_min}); // overflow is defined for atomic
static_assert(fetch_add(int_min, -1) == two_values{int_min, int_max});

// __int128
static_assert(fetch_add(__int128{42}, __int128{10}) == two_values{__int128{42}, __int128{52}});
static_assert(fetch_add(__int128{int_max}, __int128{1}) == two_values{__int128{int_max}, __int128{int_max} + __int128{1}}); // it's much bigger than 64bit

// floats
static_assert(fetch_add(10.3, 2.1) == two_values{10.3, 12.4});
static_assert(fetch_add(10.3f, 2.1f) == two_values{10.3f, 12.4f});

// pointers
static_assert(fetch_add(&array[0], 1) == two_values{&array[0], &array[1]});
static_assert(fetch_add(&small_array[0], 1) == two_values{&small_array[0], &small_array[1]});
static_assert(fetch_add(&array[1], 0) == two_values{&array[1], &array[1]});
static_assert(fetch_add(&small_array[1], 0) == two_values{&small_array[1], &small_array[1]});
static_assert(fetch_add(&array[1], -1) == two_values{&array[1], &array[0]});
static_assert(fetch_add(&small_array[1], -1) == two_values{&small_array[1], &small_array[0]});

template <typename T, typename Y> consteval auto fetch_sub(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_sub(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

// integers
static_assert(fetch_sub(true, 1) == two_values{true, false});
static_assert(fetch_sub(0, 100) == two_values{0, -100});
static_assert(fetch_sub(100, -50) == two_values{100, 150});

static_assert(fetch_sub(int_min, 1) == two_values{int_min, int_max}); // overflow is defined for atomic
static_assert(fetch_sub(int_max, -1) == two_values{int_max, int_min});

// floats
static_assert(fetch_sub(10.3, 2.3) == two_values{10.3, 8.0});
static_assert(fetch_sub(10.3f, 2.3f) == two_values{10.3f, 8.0f});

// pointers
static_assert(fetch_sub(&array[1], 1) == two_values{&array[1], &array[0]});
static_assert(fetch_sub(&small_array[1], 1) == two_values{&small_array[1], &small_array[0]});
static_assert(fetch_sub(&array[1], 0) == two_values{&array[1], &array[1]});
static_assert(fetch_sub(&small_array[1], 0) == two_values{&small_array[1], &small_array[1]});
static_assert(fetch_sub(&array[0], -1) == two_values{&array[0], &array[1]});
static_assert(fetch_sub(&small_array[0], -1) == two_values{&small_array[0], &small_array[1]});

template <typename T, typename Y> consteval auto fetch_and(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_and(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

template <typename T, typename Y> consteval auto fetch_or(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_or(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

template <typename T, typename Y> consteval auto fetch_xor(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_xor(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

template <typename T, typename Y> consteval auto fetch_nand(T original, Y arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_nand(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

static_assert(fetch_and(0b1101u, 0b1011u) == two_values{0b1101u, 0b1001u});
static_assert(fetch_or(0b1101u, 0b1011u) == two_values{0b1101u, 0b1111u});
static_assert(fetch_xor(0b1101u, 0b1011u) == two_values{0b1101u, 0b0110u});
static_assert(fetch_nand(0b1001u, 0b1011u) == two_values{0b1001u, 0xFFFFFFF6u});

template <typename T> consteval auto fetch_min(T original, T arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_min(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

template <typename T> consteval auto fetch_max(T original, T arg) -> two_values<T> {
  _Atomic(T) av = T{original};
  const T result = __c11_atomic_fetch_max(&av, arg, __ATOMIC_RELAXED);
  return two_values<T>{result, __c11_atomic_load(&av, __ATOMIC_RELAXED)};
}

static_assert(fetch_max(10, 16) == two_values{10, 16});
static_assert(fetch_max(16, 10) == two_values{16, 16});

static_assert(fetch_min(10, 16) == two_values{10, 10});
static_assert(fetch_min(16, 10) == two_values{16, 10});

