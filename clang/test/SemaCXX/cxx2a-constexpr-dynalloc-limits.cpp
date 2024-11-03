// RUN: %clang_cc1 -std=c++20 -verify -fconstexpr-steps=1024 -Wvla %s

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

namespace GH63562 {

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

// Only run these tests on 64 bits platforms
#if __LP64__
constexpr std::size_t s = S<std::size_t>(~0UL)[42]; // expected-error {{constexpr variable 's' must be initialized by a constant expression}} \
                                           // expected-note-re@#call {{in call to 'this->alloc.allocate({{.*}})'}} \
                                           // expected-note-re@#alloc {{cannot allocate array; evaluated array bound {{.*}} is too large}} \
                                           // expected-note-re {{in call to 'S({{.*}})'}}
#endif
// Check that we do not try to fold very large arrays
std::size_t s2 = S<std::size_t>(~0UL)[42];
std::size_t s3 = S<std::size_t>(~0ULL)[42];

// We can allocate and initialize a small array
constexpr std::size_t ssmall = S<std::size_t>(100)[42];

// We can allocate this array but we hikt the number of steps
constexpr std::size_t s4 = S<std::size_t>(1024)[42]; // expected-error {{constexpr variable 's4' must be initialized by a constant expression}} \
                                   // expected-note@#construct {{constexpr evaluation hit maximum step limit; possible infinite loop?}} \
                                   // expected-note@#construct_call {{in call}} \
                                   // expected-note {{in call}}



constexpr std::size_t s5 = S<std::size_t>(1025)[42]; // expected-error{{constexpr variable 's5' must be initialized by a constant expression}} \
                                   // expected-note@#alloc {{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024); use '-fconstexpr-steps' to increase this limit}} \
                                   // expected-note@#call {{in call to 'this->alloc.allocate(1025)'}} \
                                   // expected-note {{in call}}


// Check we do not perform constant initialization in the presence
// of very large arrays (this used to crash)

template <auto N>
constexpr int stack_array() {
    [[maybe_unused]] char BIG[N] = {1};  // expected-note  3{{cannot allocate array; evaluated array bound 1025 exceeds the limit (1024); use '-fconstexpr-steps' to increase this limit}}
    return BIG[N-1];
}

int a = stack_array<~0U>();
int c = stack_array<1024>();
int d = stack_array<1025>();
constexpr int e = stack_array<1024>();
constexpr int f = stack_array<1025>(); // expected-error {{constexpr variable 'f' must be initialized by a constant expression}} \
                                       //  expected-note {{in call}}
void ohno() {
  int bar[stack_array<1024>()];
  int foo[stack_array<1025>()]; // expected-warning {{variable length arrays are a C99 feature}} \
                                // expected-note {{in call to 'stack_array<1025>()'}}

  constexpr int foo[stack_array<1025>()]; // expected-warning {{variable length arrays are a C99 feature}} \
                                          // expected-error {{constexpr variable cannot have non-literal type 'const int[stack_array<1025>()]'}} \
                                          // expected-note {{in call to 'stack_array<1025>()'}}
}

}
