// RUN: %clang_cc1 -std=c++20 -verify=ref,both      -fconstexpr-steps=1024 -Wvla %s
// RUN: %clang_cc1 -std=c++20 -verify=both,both -fconstexpr-steps=1024 -Wvla %s -fexperimental-new-constant-interpreter




namespace std {
  using size_t = decltype(sizeof(0));
}

void *operator new(std::size_t, void *p) { return p; }

namespace std {
  template<typename T> struct allocator {
    constexpr T *allocate(size_t N) {
      return (T*)operator new(sizeof(T) * N); // #alloc
    }
    constexpr void deallocate(void *p) {
      operator delete(p);
    }
  };
  template<typename T, typename ...Args>
  constexpr void construct_at(void *p, Args &&...args) { // #construct
    new (p) T((Args&&)args...);
  }
}

template <typename T>
struct S {
    constexpr S(unsigned long long N)
    : data(nullptr){
        data = alloc.allocate(N);  // #call
        for(std::size_t i = 0; i < N; i ++)
            std::construct_at<T>(data + i, i); // #construct_call
    }
    constexpr T operator[](std::size_t i) const {
      return data[i];
    }

    constexpr ~S() {
        alloc.deallocate(data);
    }
    std::allocator<T> alloc;
    T* data;
};

#if __LP64__
constexpr std::size_t s = S<std::size_t>(~0UL)[42]; // both-error {{constexpr variable 's' must be initialized by a constant expression}} \
                                           // both-note-re@#call {{in call to 'this->alloc.allocate({{.*}})'}} \
                                           // both-note-re@#alloc {{cannot allocate array; evaluated array bound {{.*}} is too large}} \
                                           // both-note-re {{in call to 'S({{.*}})'}}
#endif

constexpr std::size_t ssmall = S<std::size_t>(100)[42];

constexpr std::size_t s5 = S<std::size_t>(1025)[42]; // both-error {{constexpr variable 's5' must be initialized by a constant expression}} \
                                   // both-note@#alloc {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}} \
                                   // both-note@#alloc {{use -fconstexpr-steps}} \
                                   // both-note@#call {{in call to 'this->alloc.allocate(1025)'}} \
                                   // both-note {{in call}}



template <auto N>
constexpr int stack_array() {
    [[maybe_unused]] char BIG[N] = {1};  // both-note {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}} \
                                         // both-note {{use -fconstexpr-steps}}
    return BIG[N-1];
}

int c = stack_array<1024>();
int d = stack_array<1025>();
constexpr int e = stack_array<1024>();
constexpr int f = stack_array<1025>(); // both-error {{constexpr variable 'f' must be initialized by a constant expression}} \
                                       // both-note {{in call}}

namespace GH173728 {
struct T {};

int stmt_expr() { return 1 + ({ T s[0xFFFFFFFFu][0]; 0x97 < 10000; }); }
#if __SIZEOF_SIZE_T__ == 8
int stmt_expr_truncated() {
  return 1 + ({ T s[(1ULL << 33) - 1][0]; 0x97 < 10000; });
}
#endif

template <auto N>
constexpr int default_construct() {
  T s[N][0]; // #gh173728-construct
  return 0;
}

constexpr int construct_ok = default_construct<1024>();
constexpr int construct_limit = default_construct<1025>(); // both-error {{constexpr variable 'construct_limit' must be initialized by a constant expression}} \
                                                           // both-note {{in call}}
// both-note@#gh173728-construct {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}}
// both-note@#gh173728-construct {{use -fconstexpr-steps}}

#if __SIZEOF_SIZE_T__ == 8
constexpr int construct_huge = default_construct<(1ULL << 33) - 1>(); // both-error {{constexpr variable 'construct_huge' must be initialized by a constant expression}} \
                                                                      // ref-note {{in call}}
// ref-note@#gh173728-construct {{cannot allocate array; evaluated array bound 8589934591 is too large}}
#endif

template <typename A>
constexpr int capture_copy(const A &a) {
  return [a] { return 0; }(); // #gh173728-capture
}

constexpr T src_ok[1024][0] = {};
constexpr T src_limit[1025][0] = {};
constexpr int capture_ok = capture_copy(src_ok);
constexpr int capture_limit = capture_copy(src_limit); // both-error {{constexpr variable 'capture_limit' must be initialized by a constant expression}} \
                                                       // both-note {{in call}}
// both-note@#gh173728-capture {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024)}}
// both-note@#gh173728-capture {{use -fconstexpr-steps}}
}
