template<typename T> struct Foo;

typedef Foo<char> Bar;

template<typename T> struct [[clang::preferred_name(Bar)]] Foo {};

template <typename T> struct Baz { Foo<char> member; };
