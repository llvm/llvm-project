template<typename T> struct Foo;

template<typename T>
using Bar = Foo<T>;

template<typename T> struct [[clang::preferred_name(Bar<T>)]] Foo {};

template <typename T> struct Baz { Foo<char> member; };
