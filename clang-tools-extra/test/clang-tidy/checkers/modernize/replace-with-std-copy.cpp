// RUN: %check_clang_tidy %s modernize-replace-with-std-copy %t -- \
// RUN:   -config='{CheckOptions: {modernize-replace-with-std-copy.FlagMemcpy: true}}'

namespace std {
using size_t = decltype(sizeof(int));

using int64_t = long long ;
using int32_t = int ;
using int16_t = short ;
using int8_t = char ;

template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T, typename A>
struct basic_string {
  typedef basic_string<C, T, A> _Type;
  using value_type = C;
  basic_string();
  basic_string(const C* p, const A& a = A());

  const C* data() const;
  C* data();
  size_t size() const;
};

using string = basic_string<char, std::char_traits<char>, std::allocator<char>>;
using wstring = basic_string<wchar_t, std::char_traits<wchar_t>,std::allocator<wchar_t>>;
using u16string = basic_string<char16_t, std::char_traits<char16_t>, std::allocator<char16_t>>;
using u32string = basic_string<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>>;

template <typename CharT>
class basic_string_view {
public:
  basic_string_view();

  basic_string_view(const CharT *);

  basic_string_view(const CharT *, size_t);

  basic_string_view(const basic_string_view &);

  basic_string_view &operator=(const basic_string_view &);

  const CharT* data() const;

  size_t size() const;
};

using string_view = basic_string_view<char>;


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

template <typename T, size_t N>
struct array {
  T* begin();
  T* end();
  const T* begin() const;
  const T* end() const;
  
  size_t size() const;

  const T* data() const;
  T* data();

  T _data[N];
};

template<typename T>
struct span {
  template<size_t N>
  span(T (&arr)[N]);

  const T* data() const;
  T* data();

  using value_type = T;
};

template<typename Container>
typename Container::value_type* data(Container& Arg)  {
    return Arg.data();
}

template<typename Container>
const typename Container::value_type* data(const Container& Arg) {
    return Arg.data();
}

template<typename T, size_t N>
constexpr T* data(T (&arr)[N]) {
  return arr;
}

template< class T, std::size_t N >
constexpr std::size_t size( const T (&array)[N] ) {
  return N;
}

template<typename Container>
size_t size(const Container& c) {
  return c.size();
}

wchar_t* wmemcpy( wchar_t* dest, const wchar_t* src, size_t count );
wchar_t* wmemmove( wchar_t* dest, const wchar_t* src, size_t count );

template <typename T>
struct unique_ptr {
  unique_ptr(T);
  T &operator*() const;
};
} // namespace std

namespace {
using size_t = std::size_t;

void *memcpy(void *__restrict dest, const void *__restrict src, size_t n) {
  return nullptr;
}
void *memmove(void *__restrict dest, const void *__restrict src, size_t n) {
  return nullptr;
}
wchar_t* wmemcpy( wchar_t* dest, const wchar_t* src, size_t count ) {
  return nullptr;
}
wchar_t* wmemmove( wchar_t* dest, const wchar_t* src, size_t count ) {
  return nullptr;
}
} // namespace

namespace {
void notSupportedEx() {

  { // pointee is not a collection
    char Source[] = "once upon a daydream...";
    auto *PrimitiveDest = new std::int8_t;
  
    std::memcpy(PrimitiveDest, Source, sizeof PrimitiveDest);
  }

  { // reinterpretation
    double D = 0.1;
    std::int64_t N;
    // don't warn on calls over non-sequences
    std::memcpy(&N, &D, sizeof D);
  }

  { // [de]serialization
    struct ComplexObj {
      int A;
      float B;
    };
    auto *Obj = new ComplexObj();

    char Src[sizeof(ComplexObj)] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    std::memcpy(Obj, Src, sizeof(ComplexObj));
    char Dst[sizeof(ComplexObj)];
    std::memcpy(Dst, Obj, sizeof(ComplexObj));
  }

  { // incomplete array type (should be treated the same as a raw ptr)
    char Src[] = "once upon a daydream...";
    auto CopySrc = [&Src](char Dst[], size_t sz) {
      std::memcpy(Dst, Src, sz);
    };

    char Dst[4];
    CopySrc(Dst, 4);
  }

  { // pointers
    const char *Src = "once upon a daydream...";
    char *Dst = new char[64];

    std::memcpy(Dst, Src, 24);
  }
}

void supportedEx() {
  { // two wchar c-arrays
    wchar_t Src[] = L"once upon a daydream...";
    wchar_t Dst[4];

    std::wmemcpy(Dst, Src, sizeof(Dst) / sizeof(wchar_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Dst), std::begin(Dst));

    std::wmemcpy(Dst, Src, sizeof(Dst) / sizeof(wchar_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Dst), std::begin(Dst));

    wmemcpy(Dst, Src, std::size(Src));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Src), std::begin(Dst));

    wmemmove(Dst, Src, std::size(Src));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Src), std::begin(Dst));
  }

  { // std::string + std::vector
    std::string Src = "once upon a daydream...";
    std::vector<char> Dst(4);

    std::memcpy(Dst.data(), Src.data(), Dst.size());
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), Dst.size(), std::begin(Dst));
  }

  { // std::string + std::vector
    std::string Src = "once upon a daydream...";
    std::vector<char> Dst(4);

    std::memmove(Dst.data(), Src.data(), Src.size());
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), Src.size(), std::begin(Dst));
  }

  { // reference to const c-array
    const char Src[] = "once upon a daydream...";
    char Dst[7];
    decltype(Dst)& RefDst = Dst;

    memcpy(RefDst, Src, sizeof(RefDst));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(RefDst), std::begin(RefDst));
  }

  { // std::wstring + std::unique_ptr
    std::wstring Src = L"once upon a daydream...";
    std::unique_ptr<wchar_t[7]> Dst(new wchar_t[7]);

    std::wmemcpy(*Dst, Src.data(), sizeof(*Dst) / sizeof(wchar_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(*Dst), std::begin(*Dst));
  }

  { // std::string_view + std::array
    std::string_view Src = "once upon a daydream...";
    std::array<char, 16> Dst;

    std::memmove(Dst.data(), Src.data(), Dst.size() - 1);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), Dst.size() - 1, std::begin(Dst));
  }

  { // using namespace std;
    using namespace std;
    string_view Src = "once upon a daydream...";
    array<char, 16> Dst;

    memmove(Dst.data(), Src.data(), Src.size() - 2);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), Src.size() - 2, std::begin(Dst));
  }

  { // various size arg cases
    std::int32_t Data[] = {1, 2, 3, 4, 1, 2, 3, 4};
    std::span<std::int32_t> Src{Data};
    std::vector<std::int32_t> Dst(8);

    memcpy(Dst.data(), Src.data(), Dst.size() * sizeof(std::int32_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), Dst.size(), std::begin(Dst));

    memmove(std::data(Dst), std::data(Src), sizeof(int) * std::size(Dst));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Dst), std::begin(Dst));  

    // If both arguments of the product match sizeOfExpr then the right one is lifted
    std::memcpy(std::data(Dst), std::data(Src), sizeof(int) * sizeof(std::int32_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), sizeof(int), std::begin(Dst));

    std::memcpy(std::data(Dst), std::data(Src), sizeof(decltype(Dst)::value_type) * 3);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), 3, std::begin(Dst));
  }

  { // default sizearg path
    std::vector<std::int32_t> Src(64);
    std::u32string Dst = U"once upon a daydream...";

    // Don't factor out sizeof expressions that are unlikely to be related to the value type,
    // this might lose semantic info
    memmove(std::data(Dst), std::data(Src), sizeof(float) * 3);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), (sizeof(float) * 3) / sizeof(char32_t), std::begin(Dst));
  
    // the sizeof(value_type) is nested too deep in the product expression,
    // so currently the check will leave it as is
    memmove(Dst.data(), Src.data(), 1 * 2 * sizeof(char32_t) * 4);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memmove'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), (1 * 2 * sizeof(char32_t) * 4) / sizeof(char32_t), std::begin(Dst));

    std::memcpy(Dst.data(), Src.data(), sizeof(char32_t) + sizeof(int));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), (sizeof(char32_t) + sizeof(int)) / sizeof(char32_t), std::begin(Dst));
  }

  { // SizeOfDivSizeOf - width = char
    const char Src[] = "once upon a daydream...";
    std::string DstStr = "    ";

    // superfluous, can be simplified
    std::memcpy(DstStr.data(), std::data(Src), sizeof(Src) / sizeof(char));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Src), std::begin(DstStr));
  }

  { // SizeOfDivSizeOf - width = int
    const int Src[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> DstVec(8);

    // this is probably a bug, but the check prefers not to change behaviour
    std::memcpy(DstVec.data(), std::data(Src), sizeof(Src) / sizeof(int));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'memcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), (sizeof(Src) / sizeof(int)) / sizeof(int), std::begin(DstVec));
  }

  { // SizeOfDivSizeOf - width = wchar_t
    const wchar_t Src[] = L"once upon a daydream...";
    std::wstring DstStr = L"    ";

    // simplifies
    std::wmemcpy(DstStr.data(), std::data(Src), sizeof(Src) / sizeof(wchar_t));
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: prefer std::copy_n to 'wmemcpy'
    // CHECK-FIXES: std::copy_n(std::cbegin(Src), std::size(Src), std::begin(DstStr));
  }
}
} // namespace