// RUN: %clang_cc1 -std=c++2c -verify %s

struct NotAPack;
template <typename T, auto V, template<typename> typename Tp>
void not_pack() {
    int i = 0;
    i...[0]; // expected-error {{i does not refer to the name of a parameter pack}}
    V...[0]; // expected-error {{V does not refer to the name of a parameter pack}}
    NotAPack...[0] a; // expected-error{{'NotAPack' does not refer to the name of a parameter pack}}
    T...[0] b;   // expected-error{{'T' does not refer to the name of a parameter pack}}
    Tp...[0] c; // expected-error{{'Tp' does not refer to the name of a parameter pack}}
}

template <typename T, auto V, template<typename> typename Tp>
void not_pack_arrays() {
    NotAPack...[0] a[1]; // expected-error{{'NotAPack' does not refer to the name of a parameter pack}}
    T...[0] b[1];   // expected-error{{'T' does not refer to the name of a parameter pack}}
    Tp...[0] c[1]; // expected-error{{'Tp' does not refer to the name of a parameter pack}}
}

template <typename T>
struct TTP;

void test_errors() {
    not_pack<int, 0, TTP>();
    not_pack_arrays<int, 0, TTP>();
}

namespace invalid_indexes {

int non_constant_index(); // expected-note 2{{declared here}}

template <int idx>
int params(auto... p) {
    return p...[idx]; // #error-param-size
}

template <auto N, typename...T>
int test_types() {
    T...[N] a; // #error-type-size
}

void test() {
    params<0>();   // expected-note{{here}} \
                   // expected-error@#error-param-size {{invalid index 0 for pack p of size 0}}
    params<1>(0);  // expected-note{{here}} \
                   // expected-error@#error-param-size {{invalid index 1 for pack p of size 1}}
    params<-1>(0); // expected-note{{here}} \
                   // expected-error@#error-param-size {{invalid index -1 for pack p of size 1}}

    test_types<-1>(); //expected-note {{in instantiation}} \
                      // expected-error@#error-type-size {{invalid index -1 for pack 'T' of size 0}}
    test_types<-1, int>(); //expected-note {{in instantiation}} \
                      // expected-error@#error-type-size {{invalid index -1 for pack 'T' of size 1}}
    test_types<0>(); //expected-note {{in instantiation}} \
                    // expected-error@#error-type-size {{invalid index 0 for pack 'T' of size 0}}
    test_types<1, int>(); //expected-note {{in instantiation}}  \
                         // expected-error@#error-type-size {{invalid index 1 for pack 'T' of size 1}}
}

void invalid_indexes(auto... p) {
    p...[non_constant_index()]; // expected-error {{array size is not a constant expression}}\
                                // expected-note {{cannot be used in a constant expression}}

    const char* no_index = "";
    p...[no_index]; // expected-error {{value of type 'const char *' is not implicitly convertible}}
}

void invalid_index_types() {
    []<typename... T> {
        T...[non_constant_index()] a;  // expected-error {{array size is not a constant expression}}\
                                       // expected-note {{cannot be used in a constant expression}}
    }(); //expected-note {{in instantiation}}
}

}

template <typename T, typename U>
constexpr bool is_same = false;

template <typename T>
constexpr bool is_same<T, T> = true;

template <typename T>
constexpr bool f(auto&&... p) {
    return is_same<T, decltype(p...[0])>;
}

void g() {
    int a = 0;
    const int b = 0;
    static_assert(f<int&&>(0));
    static_assert(f<int&>(a));
    static_assert(f<const int&>(b));
}

template <auto... p>
struct check_ice {
    enum e {
        x = p...[0]
    };
};

static_assert(check_ice<42>::x == 42);

struct S{};
template <auto... p>
constexpr auto constant_initializer = p...[0];
constexpr auto InitOk = constant_initializer<S{}>;

consteval int evaluate(auto... p) {
    return p...[0];
}
constexpr int x = evaluate(42, S{});
static_assert(x == 42);


namespace splice {
template <auto ... Is>
struct IL{};

template <typename ... Ts>
struct TL{};

template <typename Tl, typename Il>
struct SpliceImpl;

template <typename ... Ts, auto ...Is>
struct SpliceImpl<TL<Ts...>, IL<Is...>>{
    using type = TL<Ts...[Is]...>;
};

template <typename Tl, typename Il>
using Splice = typename SpliceImpl<Tl, Il>::type;
using type = Splice<TL<char, short, long, double>, IL<1, 2>>;
static_assert(is_same<type, TL<short, long>>);
}


namespace GH81697 {

template<class... Ts> struct tuple {
    int __x0;
};

template<auto I, class... Ts>
Ts...[I]& get(tuple<Ts...>& t) {
  return t.__x0;
}

void f() {
  tuple<int> x;
  get<0>(x);
}

}
