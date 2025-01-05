// RUN: %check_clang_tidy %s modernize-replace-with-std-copy %t -- \
// RUN:   -config='{CheckOptions: {modernize-replace-with-std-copy.FlagMemcpy: true}}'

// possible call scenarios, infeasible to cover all
// replacement type:
// [ ] members when possible
// [X] all free
// source:
//   type:
//   [ ] T[]
//     std::data:
//     [ ] free
//     [ ] N/A
//   [ ] container
//     std::data:
//     [ ] free
//     [ ] member
//     type:
//       [ ] std::vector<T>
//       [ ] std::span<T>
//       [ ] std::deque<T>
//       [ ] std::array<T, _>
//       [ ] std::string
//       [ ] std::string_view
// dest:
//   type:
//   [ ] T[]
//     std::data:
//     [ ] free
//     [ ] N/A
//   [ ] container
//     std::data:
//     [ ] free
//     [ ] member
//     type:
//       [ ] std::vector<T>
//       [ ] std::span<T>
//       [ ] std::deque<T>
//       [ ] std::array<T, _>
//       [ ] std::string
//       [ ] std::string_view
// callee:
//   name:
//   [ ] memmove
//   [ ] memcpy
//   [ ] wmemmove
//   [ ] wmemcpy
//   qualified:
//   [ ] y
//   [ ] n

// CHECK-FIXES: #include <algorithm>

namespace {

using size_t = decltype(sizeof(int));

namespace std {
using int64_t = long long ;
using int32_t = int ;
using int16_t = short ;
using int8_t = char ;

using char32 = int32_t;
using char16 = int16_t; 
using char8 = int8_t;

template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  basic_string();
  basic_string(const C* p, const A& a = A());

  const C* c_str() const;
  const C* data() const;

  _Type& append(const C* s);
  _Type& append(const C* s, size_t n);
  _Type& assign(const C* s);
  _Type& assign(const C* s, size_t n);

  int compare(const _Type&) const;
  int compare(const C* s) const;
  int compare(size_t pos, size_t len, const _Type&) const;
  int compare(size_t pos, size_t len, const C* s) const;

  size_t find(const _Type& str, size_t pos = 0) const;
  size_t find(const C* s, size_t pos = 0) const;
  size_t find(const C* s, size_t pos, size_t n) const;

  _Type& insert(size_t pos, const _Type& str);
  _Type& insert(size_t pos, const C* s);
  _Type& insert(size_t pos, const C* s, size_t n);

  _Type& operator+=(const _Type& str);
  _Type& operator+=(const C* s);
  _Type& operator=(const _Type& str);
  _Type& operator=(const C* s);
};

using string = basic_string<char, std::char_traits<char>, std::allocator<char>>;
using wstring = basic_string<wchar_t, std::char_traits<wchar_t>,std::allocator<wchar_t>>;
using u16string = basic_string<char16, std::char_traits<char16>, std::allocator<char16>>;
using u32string = basic_string<char32, std::char_traits<char32>, std::allocator<char32>>;



void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
void *memmove(void *__restrict dest, const void *__restrict src, size_t n);

template <typename T> struct vector {
  vector(size_t);

  T *data();
  const T *data() const;
  size_t size() const;
  void resize(size_t);
  using value_type = T;
};

template<typename T>
T* data(vector<T>&);

template<typename T>
T* data(T[]);

template<typename T>
const T* data(const vector<T>&);

template<typename T>
const T* data(const T[]);

size_t size(void *);

size_t strlen(const char *);
} // namespace std

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n);
void *memmove(void *__restrict dest, const void *__restrict src, size_t n);
} // namespace


namespace {
void notSupportedEx() {
  char Source[] = "once upon a daydream...", Dest[4];

  auto *PrimitiveDest = new std::int8_t;
  std::memcpy(PrimitiveDest, Source, sizeof PrimitiveDest);

  auto *PrimitiveDest2 = new std::int16_t;
  std::memcpy(PrimitiveDest2, Source, sizeof PrimitiveDest);
  std::memcpy(PrimitiveDest2, Source, sizeof PrimitiveDest2);

  double D = 0.1;
  std::int64_t N;
  // don't warn on calls over non-sequences
  std::memcpy(&N, &D, sizeof D);

  // object creation in destination buffer
  struct StructType {
    int X{42};
    void *operator new(size_t, void *) noexcept { return nullptr; }
  } Struct;
  alignas(StructType) char Buf[sizeof(StructType)];
  StructType *Ps = new (Buf) StructType; // placement new
  // // don't warn on calls over non-sequences
  std::memcpy(Ps, &Struct, sizeof Struct);

  const char *PtrSource = "once upon a daydream...";
  char *PtrDest = new char[4];
  std::memcpy(Dest, PtrSource, sizeof Dest);
  std::memcpy(PtrDest, Source, 4);
}

void noFixItEx() {
  {
    int Source[10];
    std::vector<char> Dest(5);

    memmove(Source, std::data(Dest), 1);
    // CHECK-MESSAGES: [[@LINE-1]]:20: warning: prefer std::copy_n to memcpy
    std::memmove(Source, std::data(Dest), 1);
    memmove(std::data(Source), std::data(Dest), 1);
    std::memmove(std::data(Source), std::data(Dest), 1);
    memmove(std::data(Source), std::data(Dest), 1);
    std::memmove(std::data(Source), std::data(Dest), 1);
  }
  char Source[] = "once upon a daydream...", Dest[4];

  // no FixIt when return value is used

  [](auto){}(std::memcpy(Dest, Source, sizeof Dest));
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: prefer std::copy_n to memcpy

  std::vector<std::int16_t> VecI16(4);
  // not a supported type, should be a sequence of bytes, otherwise it is difficult to compute the n in copy_n
  std::memcpy(VecI16.data(), Source,
              VecI16.size() * sizeof(decltype(VecI16)::value_type));
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy

  std::memcpy(std::data(VecI16), Source,
              VecI16.size() * sizeof(decltype(VecI16)::value_type));
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

  {
    char Source[] = "once upon a daydream...";
    char Dest[4];

    std::memcpy(Dest, Source, sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    std::memcpy(std::data(Dest), Source, sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    std::memcpy(Dest, std::data(Source), sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    std::memcpy(std::data(Dest), std::data(Source), sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    // __jm__ warn on global call as well
    memcpy(Dest, Source, sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    memcpy(std::data(Dest), Source, sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    memcpy(Dest, std::data(Source), sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    memcpy(std::data(Dest), std::data(Source), sizeof Dest);
    // CHECK-MESSAGES: [[@LINE-1]]:3: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));
  }

  {
    char Source[] = "once upon a daydream...";
    std::vector<char> Dest(4);

    std::memcpy(Dest.data(), Source, Dest.size());
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));

    std::memcpy(std::data(Dest), Source, Dest.size());
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Dest), std::begin(Dest));
  }

  {
    std::vector<char> Source(10);
    char Dest[4];

    std::memcpy(Dest, Source.data(), Source.size());
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Source), std::begin(Dest));
    std::memcpy(Dest, Source.data(), sizeof(Dest));
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(Source.data(), std::size(Dest), std::begin(Dest));
    std::memcpy(Dest, Source.data(), std::size(Dest));
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Source), std::begin(Source));

    std::memcpy(Dest, Source.data(), 1 + Source.size() / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), 1 + std::size(Source) / 2, std::begin(Dest));
    std::memcpy(Dest, Source.data(), 1 + sizeof(Dest) / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(Source.data(), 1 + std::size(Dest) / 2, std::begin(Dest));
    std::memcpy(Dest, Source.data(), 1 + std::size(Dest) / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), 1 + std::size(Dest) / 2, std::begin(Source));

    std::memcpy(Dest, std::data(Source), Source.size());
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Source), std::begin(Dest));
    std::memcpy(Dest, std::data(Source), sizeof(Dest));
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(Source.data(), std::size(Dest), std::begin(Dest));
    std::memcpy(Dest, std::data(Source), std::size(Dest));
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), std::size(Source), std::begin(Source));

    std::memcpy(Dest, std::data(Source), 1 + Source.size() / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), 1 + std::size(Source) / 2, std::begin(Dest));
    std::memcpy(Dest, std::data(Source), 1 + sizeof(Dest) / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(Source.data(), 1 + std::size(Dest) / 2, std::begin(Dest));
    std::memcpy(Dest, std::data(Source), 1 + std::size(Dest) / 2);
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: prefer std::copy_n to memcpy
    // CHECK-FIXES: std::copy_n(std::begin(Source), 1 + std::size(Dest) / 2, std::begin(Source));
  }

  // __jm__ make configurable whether stl containers should use members or free fns.
  // __jm__ for now use free fns. only

}
} // namespace