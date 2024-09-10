// RUN: %clang_cc1 -std=c++20  -Wno-all  -Wunsafe-buffer-usage-in-container -verify %s

namespace std {
  template <class T> class span {
  public:
    constexpr span(T *, unsigned){}

    template<class Begin, class End>
    constexpr span(Begin first, End last){}

    T * data();

    constexpr span() {};

    constexpr span(const std::span<T> &span) {};

    template<class R>
    constexpr span(R && range){};
  };


  template< class T >
  T&& move( T&& t ) noexcept;
}

namespace irrelevant_constructors {
  void non_two_param_constructors() {
    class Array {
    } a;
    std::span<int> S;      // no warn
    std::span<int> S1{};   // no warn
    std::span<int> S2{std::move(a)};  // no warn
    std::span<int> S3{S2};  // no warn
  }
} // irrelevant_constructors

namespace construct_wt_ptr_size {
  std::span<int> warnVarInit(int *p) {
    std::span<int> S{p, 10};                     // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S1(p, 10);                    // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S2 = std::span{p, 10};        // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S3 = std::span(p, 10);        // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S4 = std::span<int>{p, 10};   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S5 = std::span<int>(p, 10);   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S6 = {p, 10};                 // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto S7 = std::span<int>{p, 10};             // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto S8 = std::span<int>(p, 10);             // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    const auto &S9 = std::span<int>{p, 10};      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto &&S10 = std::span<int>(p, 10);          // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

#define Ten 10

    std::span S11 = std::span<int>{p, Ten};      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    if (auto X = std::span<int>{p, Ten}; S10.data()) { // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    }

    auto X = warnVarInit(p); // function return is fine
    return S;
  }

  template<typename T>
  void foo(const T &, const T &&, T);

  std::span<int> warnTemp(int *p) {
    foo(std::span<int>{p, 10},                       // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
	std::move(std::span<int>{p, 10}),            // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
	std::span<int>{p, 10});                      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    std::span<int> Arr[1] = {std::span<int>{p, 10}}; // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    if (std::span<int>{p, 10}.data()) {              // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    }
    return std::span<int>{p, 10};                    // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  }

  void notWarnSafeCases(unsigned n, int *p) {
    int X;
    unsigned Y = 10;
    std::span<int> S = std::span{&X, 1}; // no-warning
    int Arr[10];
    typedef int TenInts_t[10];
    TenInts_t Arr2;

    S = std::span{&X, 2};                // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    S = std::span{new int[10], 10};      // no-warning
    S = std::span{new int[n], n};        // no-warning
    S = std::span{new int, 1};           // no-warning
    S = std::span{new int, X};           // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    S = std::span{new int[n--], n--};    // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    S = std::span{new int[10], 11};      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    S = std::span{new int[10], 9};       // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}  // not smart enough to tell its safe
    S = std::span{new int[10], Y};       // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}  // not smart enough to tell its safe
    S = std::span{Arr, 10};              // no-warning
    S = std::span{Arr2, 10};             // no-warning
    S = std::span{Arr, Y};               // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}  // not smart enough to tell its safe
    S = std::span{p, 0};                 // no-warning
  }
} // namespace construct_wt_ptr_size

namespace construct_wt_begin_end {
  class It {};

  std::span<int> warnVarInit(It &First, It &Last) {
    std::span<int> S{First, Last};                     // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S1(First, Last);                    // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S2 = std::span<int>{First, Last};   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S3 = std::span<int>(First, Last);   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S4 = std::span<int>{First, Last};   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S5 = std::span<int>(First, Last);   // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    std::span<int> S6 = {First, Last};                 // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto S7 = std::span<int>{First, Last};             // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto S8 = std::span<int>(First, Last);             // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    const auto &S9 = std::span<int>{First, Last};      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    auto &&S10 = std::span<int>(First, Last);          // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    if (auto X = std::span<int>{First, Last}; S10.data()) { // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    }

    auto X = warnVarInit(First, Last); // function return is fine
    return S;
  }

  template<typename T>
  void foo(const T &, const T &&, T);

  std::span<int> warnTemp(It &First, It &Last) {
    foo(std::span<int>{First, Last},                       // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
	std::move(std::span<int>{First, Last}),            // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
	std::span<int>{First, Last});                      // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    std::span<int> Arr[1] = {std::span<int>{First, Last}}; // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}

    if (std::span<int>{First, Last}.data()) {              // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
    }
    return std::span<int>{First, Last};                    // expected-warning{{the two-parameter std::span construction is unsafe as it can introduce mismatch between buffer size and the bound information}}
  }
} // namespace construct_wt_begin_end

namespace test_flag {
  void f(int *p) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // this flag turns off every unsafe-buffer warning
    std::span<int> S{p, 10};   // no-warning
    p++;                       // no-warning
#pragma clang diagnostic pop

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage-in-container"
    // turn on all unsafe-buffer warnings except for the ones under `-Wunsafe-buffer-usage-in-container`
    std::span<int> S2{p, 10};   // no-warning

    p++; // expected-warning{{unsafe pointer arithmetic}}\
	    expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}
#pragma clang diagnostic pop

  }
} //namespace test_flag
