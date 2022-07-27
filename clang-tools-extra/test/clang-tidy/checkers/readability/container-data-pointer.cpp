// RUN: %check_clang_tidy %s readability-container-data-pointer %t -- -- -fno-delayed-template-parsing

typedef __SIZE_TYPE__ size_t;

namespace std {
template <typename T>
struct vector {
  using size_type = size_t;

  vector();
  explicit vector(size_type);

  T *data();
  const T *data() const;

  T &operator[](size_type);
  const T &operator[](size_type) const;
};

template <typename T>
struct basic_string {
  using size_type = size_t;

  basic_string();

  T *data();
  const T *data() const;

  T &operator[](size_t);
  const T &operator[](size_type) const;
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

template <typename T>
struct is_integral;

template <>
struct is_integral<size_t> {
  static const bool value = true;
};

template <bool, typename T = void>
struct enable_if { };

template <typename T>
struct enable_if<true, T> {
  typedef T type;
};
}

template <typename T>
void f(const T *);

#define z (0)

void g(size_t s) {
  std::vector<unsigned char> b(s);
  f(&((b)[(z)]));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(b.data());{{$}}
}

void h() {
  std::string s;
  f(&((s).operator[]((z))));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(s.data());{{$}}

  std::wstring w;
  f(&((&(w))->operator[]((z))));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(w.data());{{$}}
}

template <typename T, typename U,
          typename = typename std::enable_if<std::is_integral<U>::value>::type>
void i(U s) {
  std::vector<T> b(s);
  f(&b[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(b.data());{{$}}
}

template void i<unsigned char, size_t>(size_t);

void j(std::vector<unsigned char> * const v) {
  f(&(*v)[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(v->data());{{$}}
}

void k(const std::vector<unsigned char> &v) {
  f(&v[0]);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}f(v.data());{{$}}
}

void l() {
  unsigned char b[32];
  f(&b[0]);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:5: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
}

template <typename T>
void m(const std::vector<T> &v) {
  return &v[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}return v.data();{{$}}
}

template <typename T>
struct container_without_data {
  using size_type = size_t;
  T &operator[](size_type);
  const T &operator[](size_type) const;
};

template <typename T>
const T *n(const container_without_data<T> &c) {
  // c has no "data" member function, so there should not be a warning here:
  return &c[0];
}

const int *o(const std::vector<std::vector<std::vector<int>>> &v, const size_t idx1, const size_t idx2) {
  return &v[idx1][idx2][0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}return v[idx1][idx2].data();{{$}}
}

std::vector<int> &select(std::vector<int> &u, std::vector<int> &v) {
  return v;
}

int *p(std::vector<int> &v1, std::vector<int> &v2) {
  return &select(*&v1, v2)[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}return select(*&v1, v2).data();{{$}}
}

int *q(std::vector<int> ***v) {
  return &(***v)[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}return (**v)->data();{{$}}
}

struct VectorHolder {
  std::vector<int> v;
};

int *r() {
  VectorHolder holder;
  return &holder.v[0];
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 'data' should be used for accessing the data pointer instead of taking the address of the 0-th element [readability-container-data-pointer]
  // CHECK-FIXES: {{^  }}return holder.v.data();{{$}}
}
