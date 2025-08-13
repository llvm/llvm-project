// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>

namespace std {
template <typename T, size_t N>
struct array {
  const T *data() const noexcept;
  T *data() noexcept;
  size_t size() const noexcept;
  const T &operator[](const size_t idx) const;
};

template <typename CharT>
struct basic_string {
  const CharT *data() const noexcept;
  CharT *data() noexcept;
  const CharT *c_str() const noexcept;
  size_t size() const noexcept;
  size_t length() const noexcept;
};

typedef basic_string<char> string;

template <typename CharT>
struct basic_string_view {
  const CharT *data() const noexcept;
  size_t size() const noexcept;
  size_t length() const noexcept;
};

typedef basic_string_view<char> string_view;

template <typename T>
struct span {
  T *data() const noexcept;
  size_t size() const noexcept;
  size_t size_bytes() const noexcept;
  span<T> first(size_t count) const noexcept;
  span<T> last(size_t count) const noexcept;
  span<T> subspan(size_t offset, size_t count) const noexcept;
  const T &operator[](const size_t idx) const;
};

template <typename T>
struct vector {
  const T *data() const noexcept;
  T *data() noexcept;
  size_t size() const noexcept;
};

}  // namespace std

template <typename T>
struct my_vec {
  const T *data() const noexcept;
  T *data() noexcept;
  size_t size() const noexcept;
};

extern "C" {

// expected-note@+1 15{{consider using a safe container and passing '.data()' to the parameter 's' and '.size()' to its dependent parameter 'len' or 'std::span' and passing '.first(...).data()' to the parameter 's'}}
void cb_char(char *__counted_by(len) s, size_t len);

void cb_cchar(const char *__counted_by(len) s, size_t len);

// expected-note@+1 3{{consider using 'std::span' and passing '.first(...).data()' to the parameter 's'}}
void cb_cchar_42(const char *__counted_by(42) s);

// expected-note@+1 19{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'count' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_int(int *__counted_by(count) p, size_t count);

// expected-note@+1 34{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'count' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_cint(const int *__counted_by(count) p, size_t count);

// expected-note@+1 10{{consider using 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_cint_42(const int *__counted_by(42) p);

// expected-note@+1 6{{consider using 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_cint_multi(const int *__counted_by((a + b) * (c - d)) p, int a, int b, int c, int d);

// expected-note@+1 13{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void sb_cvoid(const void *__sized_by(size) p, size_t size);

// expected-note@+1 {{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void sb_cchar(const char *__sized_by(size) p, size_t size);

// expected-note@+1 5{{consider using 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void sb_cvoid_42(const void *__sized_by(42) p);

// expected-note@+1 6{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void sb_cint(const int *__sized_by(size) p, size_t size);

// expected-note@+1 3{{consider using 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void sb_cint_42(const int *__sized_by(42) p);

void cb_cint_array(const int (* __counted_by(size) p)[10], size_t size);

}  // extern "C"

// Check allowed classes and method.

void allowed_class_and_method(std::array<int, 42> &a, std::string &s,
                              std::string_view sv, std::span<int> sp,
                              std::vector<int> &v) {
  cb_int(a.data(), a.size());
  cb_cint(a.data(), a.size());

  cb_char(s.data(), s.size());
  cb_char(s.data(), s.length());
  cb_cchar(s.data(), s.size());
  cb_cchar(s.data(), s.length());
  cb_cchar(s.c_str(), s.size());
  cb_cchar(s.c_str(), s.length());

  cb_cchar(sv.data(), sv.size());
  cb_cchar(sv.data(), sv.length());

  cb_cint(sp.data(), sp.size());
  sb_cvoid(sp.data(), sp.size_bytes());
  sb_cint(sp.data(), sp.size_bytes());

  cb_int(v.data(), v.size());
  cb_cint(v.data(), v.size());
}

void not_allowed_class_and_method(my_vec<int> &mv) {
  cb_int(mv.data(), mv.size()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

// Check if the object argument is the same for .data() and .size().

void object_arg(std::span<int> a, std::span<int> b) {
  cb_cint(a.data(), a.size());
  cb_cint(a.data(), b.size()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

// Check passing to pointers of different type has the correct .size() method.

static_assert(sizeof(char) == 1);

void pointers_of_diff_type(std::array<int, 42> &a, std::string &s,
                           std::span<int> sp_int, std::span<char> sp_char) {
  cb_cint(a.data(), a.size());
  sb_cvoid(a.data(), a.size()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cchar(s.data(), s.size());
  sb_cvoid(s.data(), s.size());

  cb_cint(sp_int.data(), sp_int.size());
  cb_cint(sp_int.data(), sp_int.size_bytes()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(sp_int.data(), sp_int.size());      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(sp_int.data(), sp_int.size_bytes());
  sb_cint(sp_int.data(), sp_int.size());      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(sp_int.data(), sp_int.size_bytes());

  cb_cchar(sp_char.data(), sp_char.size());
  cb_cchar(sp_char.data(), sp_char.size_bytes());
  sb_cvoid(sp_char.data(), sp_char.size());
  sb_cvoid(sp_char.data(), sp_char.size_bytes());
}

template <typename IntT, typename CharT>
void pointers_of_diff_type_uninstantiated(std::array<IntT, 42> &a, std::string &s,
                                         std::span<IntT> sp_int, std::span<CharT> sp_char) {
  cb_cint(a.data(), a.size());
  sb_cvoid(a.data(), a.size());

  cb_cchar(s.data(), s.size());
  sb_cvoid(s.data(), s.size());

  cb_cint(sp_int.data(), sp_int.size());
  cb_cint(sp_int.data(), sp_int.size_bytes());
  sb_cvoid(sp_int.data(), sp_int.size());
  sb_cvoid(sp_int.data(), sp_int.size_bytes());
  sb_cint(sp_int.data(), sp_int.size());
  sb_cint(sp_int.data(), sp_int.size_bytes());

  cb_cchar(sp_char.data(), sp_char.size());
  cb_cchar(sp_char.data(), sp_char.size_bytes());
  sb_cvoid(sp_char.data(), sp_char.size());
  sb_cvoid(sp_char.data(), sp_char.size_bytes());
}

template <typename IntT, typename CharT>
void pointers_of_diff_type_instantiated(std::array<IntT, 42> &a, std::string &s,
                                    std::span<IntT> sp_int, std::span<CharT> sp_char) {
  cb_cint(a.data(), a.size());
  sb_cvoid(a.data(), a.size()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cchar(s.data(), s.size());
  sb_cvoid(s.data(), s.size());

  cb_cint(sp_int.data(), sp_int.size());
  cb_cint(sp_int.data(), sp_int.size_bytes()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(sp_int.data(), sp_int.size());      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(sp_int.data(), sp_int.size_bytes());
  sb_cint(sp_int.data(), sp_int.size());      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(sp_int.data(), sp_int.size_bytes());

  cb_cchar(sp_char.data(), sp_char.size());
  cb_cchar(sp_char.data(), sp_char.size_bytes());
  sb_cvoid(sp_char.data(), sp_char.size());
  sb_cvoid(sp_char.data(), sp_char.size_bytes());
}

// explicitly instantiate the templated function to trigger processing of bounds safety attributes
template void pointers_of_diff_type_instantiated<int, char>(std::array<int, 42> &a, std::string &s,
                                                        std::span<int> sp_int, std::span<char> sp_char);

// Check if passing to __counted_by(const) uses a subview.

void cb_const_subview(std::string &s, std::span<int> sp) {
  cb_cchar_42(s.data());  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cchar_42(s.c_str()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cint_42(sp.data()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(42).data());
  cb_cint_42(sp.last(42).data());
  cb_cint_42(sp.subspan(0, 42).data());
}

// Check if the subview has a type with a correct size.

static_assert(sizeof(char) == 1);

void pointers_of_diff_type_subview(std::span<int> sp_int,
                                   std::span<char> sp_char) {
  cb_cint_42(sp_int.first(42).data());
  sb_cint_42(sp_int.first(42).data()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid_42(sp_int.first(42).data()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cchar_42(sp_char.first(42).data());
  sb_cvoid_42(sp_char.first(42).data());
}

// Check if subspan()'s offset doesn't matter.

void subspan_offset(std::span<int> sp, size_t x) {
  cb_cint_42(sp.subspan(0, 42).data());
  cb_cint_42(sp.subspan(1337, 42).data());
  cb_cint_42(sp.subspan(x, 42).data());
  cb_cint_42(sp.subspan(sp.size() - x, 42).data());
}

// Check count expression.

void count_expr_const(std::span<int> sp, size_t x) {
  cb_cint_42(sp.first(42).data());
  cb_cint_42(sp.first((42)).data());
  cb_cint_42(sp.first(0).data());         // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(41).data());        // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(43).data());        // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(10 + 10).data());   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(x).data());         // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(sp.size()).data()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(sp.first(42 + x).data());    // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void count_expr_simple(std::span<int> sp, size_t x) {
  cb_cint(sp.first(42).data(), 42);
  cb_cint(sp.first((42)).data(), 42);
  cb_cint(sp.first(42).data(), (42));
  cb_cint(sp.first(((42))).data(), (((42))));
  cb_cint(sp.first(x).data(), x);
  cb_cint(sp.first(42 + x).data(), 42 + x);
  cb_cint(sp.first(x * x).data(), x * x);

  cb_cint(sp.first(-1).data(), -1);         // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(42).data(), 100);        // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(x).data(), 42);          // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(42).data(), x);          // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(sp.size()).data(), x);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(x).data(), sp.size());   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(42 + x).data(), 42 - x); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint(sp.first(x * x).data(), x + x);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void count_expr_multi(std::span<int> sp, int w, int x, int y, int z) {
  cb_cint_multi(sp.first((4 + 3) * (2 - 1)).data(), 4, 3, 2, 1);
  cb_cint_multi(sp.first((w + x) * (y - z)).data(), w, x, y, z);
  cb_cint_multi(sp.first((1 + x) * (y - 2)).data(), 1, x, y, 2);
  cb_cint_multi(sp.first(((1 + 2) + (1 + w)) * ((x + 2) - (y + z))).data(), 1 + 2, 1 + w, x + 2, y + z);

  cb_cint_multi(sp.first((4 + 3) * (2 - 1)).data(), 4, 3, 2, 42);    // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_multi(sp.first((w + x) * (y - z)).data(), w, x, y, z + z); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_multi(sp.first((w + x) * (y + z)).data(), w, x, y, z);     // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

// Check passing arrays.

void const_array(size_t x) {
  int a[42], b[10];

  cb_int(a, 42);
  cb_int(a, 10);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(a, 100); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(a, x);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(a, sizeof(a)); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cint_42(a);
  cb_cint_42(b); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  sb_cvoid(a, 42); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(a, 42);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid_42(a);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint_42(a);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(a, sizeof(a)); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(a,sizeof(a)); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid((void *)a, sizeof(a));
  sb_cvoid((char *)a, sizeof(a));
  sb_cchar((char *)a, sizeof(a));

  char c[42];

  cb_char(c, 42);
  cb_char(c, 10);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_char(c, 100); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_char(c, x);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cchar_42(c);

  sb_cvoid(c, 42);
  sb_cvoid_42(c);
  sb_cvoid(c, sizeof(c));
  sb_cvoid(c, sizeof(decltype(c)));
  sb_cvoid(c, sizeof(char[42]));

}

void other_arrays(size_t n) {
  int vla[n];
  extern int incomplete_array[];
  cb_int(vla, n);              // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(incomplete_array, n); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

// Count-attributed to count-attributed.

void from_cb_int(int *__counted_by(len) p, size_t len) {
  cb_int(p, len);
  cb_cint(p, len);
  cb_cint(p, 42);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_42(p);    // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(p, len);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(p, len); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void from_cb_int_42(int *__counted_by(42) p, size_t x) {
  cb_int(p, 42);
  cb_cint(p, 42);
  cb_cint_42(p);
  cb_int(p, 10);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, 100);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, x);    // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint(p, 42);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(p, 42); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cint_42(p);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid_42(p);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void from_cb_char(char *__counted_by(len) s, size_t len) {
  cb_char(s, len);
  cb_cchar(s, len);
  sb_cvoid(s, len);
  cb_cchar_42(s); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid_42(s); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void from_sb_void(void *__sized_by(size) p, size_t size) {
  sb_cvoid(p, size);
  sb_cvoid_42(p); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void from_cb_int_multi(int *__counted_by((a + b) * (c - d)) p, int a, int b, int c, int d) {
  cb_cint_multi(p, a, b, c, d);
  cb_cint_multi(p, d, c, b, a);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_multi(p, 1, 2, 3, 4);  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_cint_multi(p, a, b, 42, d); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_int(p, (a + b) * (c - d));
  cb_int(p, (d + c) * (b - a));  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, (a - b) * (c + d));  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, (a + 42) * (c - d)); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, a + b);              // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, a);                  // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(p, 42);                 // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void nullptr_as_arg(void * __sized_by(size) p, unsigned size) { //expected-note{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'size' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
  nullptr_as_arg(nullptr, 0);
  nullptr_as_arg(nullptr, size); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

void single_variable() {
  int Var;
  int Arr[10];

  cb_int(&Var, 1);
  sb_cvoid(&Var, sizeof(int));
  sb_cvoid(&Var, sizeof(Var));
  sb_cvoid(&Var, sizeof(decltype(Var)));
  sb_cchar((char*)&Var, 1);              // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_int(&Var, 42);                      // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(&Var, sizeof(Arr));           // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(&Var, sizeof(decltype(Arr))); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  cb_cint_array(&Arr, 1);
  sb_cvoid(&Arr, sizeof(Arr));
  sb_cvoid(&Arr, sizeof(int[10]));
  sb_cvoid(&Arr, sizeof(decltype(Arr)));
  sb_cvoid(&Arr, sizeof(Var));           // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(&Arr, sizeof(int[11]));       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  sb_cvoid(&Arr, sizeof(decltype(Var))); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
}

namespace test_subscript {
  void foo(std::span<int> S) {
    std::array<int, 10> A;

    cb_cint(S.first(S[5]).data(), S[5]);
    cb_cint(S.first(A[5]).data(), A[5]);
    cb_cint(S.subspan(0, S[5]).data(), S[5]);
    cb_cint(S.subspan(0, A[5]).data(), A[5]);
    cb_cint(S.subspan(0, S[5]).data(), A[5]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.subspan(0, S[5]).data(), S[6]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

    unsigned x;

    cb_cint(S.first(S[x]).data(), S[x]);
    cb_cint(S.first(A[x]).data(), A[x]);
    cb_cint(S.subspan(0, S[x]).data(), S[x]);
    cb_cint(S.subspan(0, A[x]).data(), A[x]);
    cb_cint(S.first(S[x + 1]).data(), S[x + 1]);
    cb_cint(S.subspan(0, A[x + 1]).data(), A[x + 1]);
    cb_cint(S.first(A[A[x]]).data(), A[A[x]]);
    cb_cint(S.first(A[A[x] + 1]).data(), A[A[x] + 1]);
    cb_cint(S.first(S[x + 1]).data(), S[x - 1]);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.first(A[A[x]]).data(), A[A[x] + 1]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

    std::array<std::array<int, 10>, 10> A2d;

    cb_cint(S.first(A2d[x][5]).data(), A2d[x][5]);
    cb_cint(S.first(A2d[x][5]).data(), A2d[5][x]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  }

  struct T {
    std::array<int, 10> a;
  } t, *tp;

  struct TT {
    struct T t;
    struct T * tp;
  } tt, *ttp;

  void bar(std::span<int> S) {
    unsigned x;

    cb_cint(S.first(t.a[x]).data(), t.a[x]);
    cb_cint(S.first(tp->a[x]).data(), tp->a[x]);
    cb_cint(S.first(tt.t.a[x]).data(), tt.t.a[x]);
    cb_cint(S.first(ttp->t.a[x]).data(), ttp->t.a[x]);
    cb_cint(S.first(tt.tp->a[x]).data(), tt.tp->a[x]);
    cb_cint(S.first(ttp->tp->a[x]).data(), ttp->tp->a[x]);
    cb_cint(S.first(t.a[x]).data(), tp->a[x]);     // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.first(tt.t.a[x]).data(), tt.tp->a[x]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.first(ttp->t.a[x]).data(), ttp->tp->a[x]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

    int cArr[10];

    cb_cint(S.first(cArr[5]).data(), cArr[5]);
    cb_cint(S.first(cArr[1]).data(), cArr[sizeof(char)]);
    cb_cint(S.first(cArr[5]).data(), cArr[6]); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.first(cArr[5]).data(), cArr[sizeof(char)]);// expected-warning{{unsafe assignment to function parameter of count-attributed type}}

    std::array<struct TT, 10> ttA;

    cb_cint(S.first(ttA[4].t.a[x]).data(), ttA[4].t.a[x]);
    cb_cint(S.first(ttA[4].tp->a[x]).data(), ttA[4].tp->a[x]);
    cb_cint(S.first(ttA[3].t.a[x]).data(), ttA[4].t.a[x]);// expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S.first(ttA[3].tp->a[x]).data(), ttA[4].tp->a[x]);// expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  }
} //namespace test_subscript

namespace test_member_fun_call {
  struct T {
    unsigned size() {return 0;}                   // missing 'const'
    unsigned size(int x) const {return 0;}        // has arguments
    unsigned supported_size() const {return 0;}
  } t, *tp;

  void foo(std::span<int> S1, std::span<int> S2) {
    cb_cint(S1.first(S2.size()).data(), S2.size());
    cb_cint(S1.first(S2.size()).data(), S1.size()); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}


    cb_cint(S1.first(t.supported_size()).data(), t.supported_size());
    cb_cint(S1.first(tp->supported_size()).data(), tp->supported_size());
    cb_cint(S1.first(t.size()).data(), t.size());   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S1.first(t.size(0)).data(), t.size(0)); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_cint(S1.first(t.supported_size()).data(), tp->supported_size()); // expected-warning{{unsafe assignment to function parameter of count-attributed}}
  }
  void bar(std::span<int> S) {
    std::array<struct T, 10> Ts;
    std::array<struct T*, 10> Tps;
    unsigned x;

    cb_cint(S.first(Ts[5].supported_size()).data(), Ts[5].supported_size());
    cb_cint(S.first(Ts[x].supported_size()).data(), Ts[x].supported_size());
    cb_cint(S.first(Tps[5]->supported_size()).data(), Tps[5]->supported_size());
    cb_cint(S.first(Ts[x].supported_size()).data(), Ts[5].supported_size());  // expected-warning{{unsafe assignment to function parameter of count-attributed}}
    cb_cint(S.first(Ts[x].supported_size()).data(), Tps[x]->supported_size());// expected-warning{{unsafe assignment to function parameter of count-attributed}}
  }

  template<typename Ty, size_t N>
  struct GenT {
    unsigned size() {return 0;}                   // missing 'const'
    unsigned size(int x) const {return 0;}        // has arguments
    static unsigned static_size() {return 0;}     // static function is not supported
    unsigned supported_size() const {return 0;}

    void test(std::span<int> S) {
      cb_cint(S.first(supported_size()).data(), supported_size());
      cb_cint(S.first(supported_size()).data(), size()); // expected-warning{{unsafe assignment to function parameter of count-attributed}}
    }
  };

  void baz(std::span<int> S) {
    GenT<int, 5> T5;
    GenT<int, 10> T10;

    cb_cint(S.first(T5.supported_size()).data(), T5.supported_size());
    cb_cint(S.first(T5.static_size()).data(), T5.static_size()); // expected-warning{{unsafe assignment to function parameter of count-attributed}}
    cb_cint(S.first(GenT<int, 5>::static_size()).data(), GenT<int, 5>::static_size()); // expected-warning{{unsafe assignment to function parameter of count-attributed}}
    cb_cint(S.first(T5.supported_size()).data(), T10.supported_size()); // expected-warning{{unsafe assignment to function parameter of count-attributed}}
    T5.test(S);
  }
} // namespace test_member_fun_call

namespace test_member_access {
  class Base {
  public:
    char * __counted_by(n) p;
    size_t n;

    char * __counted_by(n + m) q;
    size_t m;

    void foo() {
      cb_char(p, n);       // no warn
      cb_char(p, m);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_char(q, n + m);   // no warn
      cb_char(q, m);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }

    void bar(size_t n) { // parameter n makes Base::n invisible
      cb_char(p, n);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_char(q, n + m);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }
  };

  class Derived : public Base {
    void foo() {
      cb_char(p, n);       // no warn
      cb_char(p, m);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_char(q, n + m);   // no warn
      cb_char(q, m);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }

    void bar(size_t n) {
      cb_char(p, n);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_char(q, n + m);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }
  };


  class Derived2 : public Base {
    size_t n;   // Derived2::n makes Base::n invisible
    void foo() {
      cb_char(p, n);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_char(q, n + m);   // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }
  };

  void memberAccessArgs(Base B, Base B2) {
    cb_char(B.p, B.n);        // no warn
    cb_char(B.p, B2.n);       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    cb_char(B.q, B.n + B.m);  // no warn
    cb_char(B.p, B.n + B2.m); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  }

  class MyClass {
    void test_brackets(int p[__counted_by(n)], size_t n, int * q[__counted_by(n)]) {
      cb_int(p, n);
      cb_int(p, 10); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
      cb_int(*q, n); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
    }
  };
} // test_member_access

namespace output_param_test {
  void cb_out_ptr(int * __counted_by(n) * p, size_t n);
  void cb_out_count(int * __counted_by(*n) p, size_t * n);
  void cb_out_both(int * __counted_by(*n) * p, size_t * n);

  void test(int * __counted_by(n) p, size_t n, int * __counted_by(*m) *q, size_t *m) {
    cb_out_ptr(&p, n);   // no error
    cb_out_ptr(&p, *&n);   // no error
    cb_out_ptr(q, *m);   // no error
    cb_out_count(p, &n);  // no error
    cb_out_count(*q, m);  // no error
    cb_out_count(*q, *&m);  // no error
    // The following pattern is a BoundsSafety error for read-only parameter 'p':
    //     cb_out_both(&p, &n);
    cb_out_both(q, m);   // no error

    size_t local_n = n;
    int * __counted_by(local_n) local_p = p;

    cb_out_ptr(&local_p, local_n);   // no error
    cb_out_count(local_p, &local_n); // no error
    cb_out_both(&local_p, &local_n); // no error
  }

  class TestClassMemberFunctions {
    void test(int * __counted_by(n) p, size_t n, int * __counted_by(*m) *q, size_t *m) {
      cb_out_ptr(&p, n);   // no error
      cb_out_ptr(&p, *&n);   // no error
      cb_out_ptr(q, *m);   // no error
      cb_out_count(p, &n);  // no error
      cb_out_count(*q, m);  // no error
      cb_out_count(*q, *&m);  // no error
      // The following pattern is a BoundsSafety error for read-only parameter 'p':
      //     cb_out_both(&p, &n);
      cb_out_both(q, m);   // no error

      size_t local_n = n;
      int * __counted_by(local_n) local_p = p;

      cb_out_ptr(&local_p, local_n);   // no error
      cb_out_count(local_p, &local_n); // no error
      cb_out_both(&local_p, &local_n); // no error
    }
  };

} // namespace output_param_test


static void previous_infinite_loop(int * __counted_by(n) p, size_t n) {
  previous_infinite_loop(p, n);
}

static void previous_infinite_loop2(int * __counted_by(n + 10) p, size_t n) {
  previous_infinite_loop2(p, n);
}

static void previous_infinite_loop3(int * __counted_by(n + n * m) p, size_t n,
// expected-note@+1 {{consider using 'std::span' and passing '.first(...).data()' to the parameter 'q'}}
				    int * __counted_by(m * m + o) q,
// expected-note@+1 {{consider using a safe container and passing '.data()' to the parameter 'r' and '.size()' to its dependent parameter 'o' or 'std::span' and passing '.first(...).data()' to the parameter 'r'}}
				    int * __counted_by(o) r, size_t m, size_t o) {
  previous_infinite_loop3(p, n, q, r, m, o);
  previous_infinite_loop3(p, n, q, r, m, o + 1); // expected-warning 2{{unsafe assignment to function parameter of count-attributed type}}
}

// Check default args.

void cb_def_arg_null_0(int *__counted_by(count) p = nullptr, size_t count = 0);

// expected-note@+1{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'count' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_def_arg_0_null(size_t count = 0, int *__counted_by(count) p = nullptr);

// expected-note@+1 6{{consider using a safe container and passing '.data()' to the parameter 'p' and '.size()' to its dependent parameter 'count' or 'std::span' and passing '.first(...).data()' to the parameter 'p'}}
void cb_def_arg_null_42(int *__counted_by(count) p = nullptr, size_t count = 42);

void default_arg(std::span<int> sp) {
  cb_def_arg_null_0();
  cb_def_arg_null_0(nullptr);
  cb_def_arg_null_0(nullptr, 0);
  cb_def_arg_null_0(sp.data(), sp.size());

  cb_def_arg_0_null();
  cb_def_arg_0_null(0);
  cb_def_arg_0_null(0, nullptr);
  cb_def_arg_0_null(sp.size(), sp.data());
  cb_def_arg_0_null(42);                       // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_def_arg_0_null(42, sp.first(42).data());

  cb_def_arg_null_42();                        // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_def_arg_null_42(nullptr);                 // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_def_arg_null_42(nullptr, 42);             // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_def_arg_null_42(nullptr, 0);
  cb_def_arg_null_42(sp.data());               // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  cb_def_arg_null_42(sp.data(), sp.size());
  cb_def_arg_null_42(sp.first(42).data());
  cb_def_arg_null_42(sp.first(42).data(), 42);
  cb_def_arg_null_42(sp.first(41).data());     // expected-warning{{unsafe assignment to function parameter of count-attributed type}}

  // Check if the diagnostic is at the right paren.
  // clang-format off
  cb_def_arg_null_42
    (
    ); // expected-warning{{unsafe assignment to function parameter of count-attributed type}}
  // clang-format on
}
