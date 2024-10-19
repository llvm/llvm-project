// RUN: %check_clang_tidy %s modernize-replace-memcpy-with-std-copy %t

// CHECK-FIXES: #include <algorithm>

namespace {

using size_t = decltype(sizeof(int));

namespace std {
typedef long long int64_t;
typedef short int16_t;
typedef char int8_t;

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);

template <typename T> struct vector {
  vector(size_t);

  T *data();
  size_t size() const;
  void resize(size_t);
  using value_type = T;
};

size_t size(void *);

size_t strlen(const char *);
} // namespace std

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
} // namespace

void notSupportedEx() {
  char source[] = "once upon a daydream...", dest[4];

  auto *primitiveDest = new std::int8_t;
  std::memcpy(primitiveDest, source, sizeof primitiveDest);

  auto *primitiveDest2 = new std::int16_t;
  std::memcpy(primitiveDest2, source, sizeof primitiveDest);
  std::memcpy(primitiveDest2, source, sizeof primitiveDest2);

  double d = 0.1;
  std::int64_t n;
  // don't warn on calls over non-sequences
  std::memcpy(&n, &d, sizeof d);

  // object creation in destination buffer
  struct S {
    int x{42};
    void *operator new(size_t, void *) noexcept { return nullptr; }
  } s;
  alignas(S) char buf[sizeof(S)];
  S *ps = new (buf) S; // placement new
  // // don't warn on calls over non-sequences
  std::memcpy(ps, &s, sizeof s);

  const char *pSource = "once upon a daydream...";
  char *pDest = new char[4];
  std::memcpy(dest, pSource, sizeof dest);
  std::memcpy(pDest, source, 4);
}

void noFixItEx() {
  char source[] = "once upon a daydream...", dest[4];

  // no FixIt when return value is used
  auto *ptr = std::memcpy(dest, source, sizeof dest);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: prefer std::copy_n to memcpy

  std::vector<std::int16_t> vec_i16(4);
  // not a supported type, should be a sequence of bytes, otherwise it is difficult to compute the n in copy_n
  std::memcpy(vec_i16.data(), source,
              vec_i16.size() * sizeof(decltype(vec_i16)::value_type));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
}

void sequenceOfBytesEx() {
  // the check should support memcpy conversion for the following types:
  // T[]
  // std::vector<T>
  // std::span<T>
  // std::deque<T>
  // std::array<T, _>
  // std::string
  // std::string_view
  // where T is byte-like

  char source[] = "once upon a daydream...", dest[4];
  std::memcpy(dest, source, sizeof dest);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(source), std::size(dest), std::begin(dest));

  // __jm__ warn on global call as well
  memcpy(dest, source, sizeof dest);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(source), std::size(dest), std::begin(dest));

  std::vector<char> vec_i8(4);
  std::memcpy(vec_i8.data(), source, vec_i8.size());
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(source), std::size(vec_i8), std::begin(vec_i8));

  // __jm__ make configurable whether stl containers should use members or free fns.
  // __jm__ for now use free fns. only

  std::memcpy(dest, vec_i8.data(), vec_i8.size());
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(vec_i8), std::size(vec_i8), std::begin(dest));
  std::memcpy(dest, vec_i8.data(), sizeof(dest));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(vec_i8.data(), std::size(dest), std::begin(dest));
  std::memcpy(dest, vec_i8.data(), std::size(dest));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(source), std::size(vec_i8), std::begin(vec_i8));

  std::memcpy(dest, vec_i8.data(), 1 + vec_i8.size() / 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(vec_i8), 1 + std::size(vec_i8) / 2, std::begin(dest));
  std::memcpy(dest, vec_i8.data(), 1 + sizeof(dest) / 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(vec_i8.data(), 1 + std::size(dest) / 2, std::begin(dest));
  std::memcpy(dest, vec_i8.data(), 1 + std::size(dest) / 2);
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
  // CHECK-FIXES: std::copy_n(std::begin(source), 1 + std::size(dest) / 2, std::begin(vec_i8));
}

// void uninitialized_copy_ex() {
//     std::vector<std::string> v = {"This", "is", "an", "example"};

//     std::string* p;
//     std::size_t sz;
//     std::tie(p, sz) = std::get_temporary_buffer<std::string>(v.size());
//     sz = std::min(sz, v.size());

//     std::uninitialized_copy_n(v.begin(), sz, p);

//     for (std::string* i = p; i != p + sz; ++i)
//     {
//         std::cout << *i << ' ';
//         i->~basic_string<char>();
//     }
//     std::cout << '\n';

//     std::return_temporary_buffer(p);
// }