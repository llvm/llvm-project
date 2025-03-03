// RUN: %check_clang_tidy %s modernize-replace-with-std-copy %t

//CHECK-FIXES: #include <algorithm>

namespace {
using size_t = decltype(sizeof(int));

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n) {
  return nullptr;
}
void *memmove(void *__restrict dest, const void *__restrict src, size_t n) {
  return nullptr;
}
} // namespace

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


namespace {
void notSupportedEx() {
    char Source[] = "once upon a daydream...";

    auto *PrimitiveDest = new std::int8_t;
    std::memmove(PrimitiveDest, Source, sizeof PrimitiveDest);

    double D = 0.1;
    std::int64_t N;
    // don't warn on calls over non-sequences
    std::memmove(&N, &D, sizeof D);

    std::vector<char> Dest(4);

    // don't warn on memcpy by default
    memcpy(Dest.data(), Source, Dest.size());
    std::memcpy(std::data(Dest), Source, Dest.size());
}

void noFixItEx() {
    // value type widths are different and so a fix is not straightforward
    int Source[5];
    std::vector<char> Dest(20);

    memmove(std::data(Dest), Source, 20);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
}

void supportedEx() {
    {
        char Source[] = "once upon a daydream...";
        char Dest[4];

        std::memmove(Dest, Source, sizeof Dest);
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: prefer std::copy_n to 'memmove'
        // CHECK-FIXES: std::copy_n(std::cbegin(Source), std::size(Dest), std::begin(Dest));

        memmove(std::data(Dest), std::data(Source), sizeof Dest);
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: prefer std::copy_n to 'memmove'
        // CHECK-FIXES: std::copy_n(std::cbegin(Source), std::size(Dest), std::begin(Dest));
    }
    
    {
        char Source[] = "once upon a daydream...";
        std::vector<char> Dest(4);

        std::memmove(Dest.data(), Source, Dest.size());
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: prefer std::copy_n to 'memmove'
        // CHECK-FIXES: std::copy_n(std::cbegin(Source), Dest.size(), std::begin(Dest));

        std::memmove(std::data(Dest), Source, Dest.size());
        // CHECK-MESSAGES: [[@LINE-1]]:9: warning: prefer std::copy_n to 'memmove'
        // CHECK-FIXES: std::copy_n(std::cbegin(Source), Dest.size(), std::begin(Dest));
    }
}
} // namespace