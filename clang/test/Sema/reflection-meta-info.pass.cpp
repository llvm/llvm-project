// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only
// RUN: %clang_cc1 %s -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only

typedef int int32_t;
using A = int;

using info = decltype(^^int);

template <auto R>
consteval auto f1() {
  return R;
}

template <decltype(^^int) R>
consteval auto f2() {
  return R;
}

consteval void test()
{
    constexpr auto r = ^^int;
    constexpr auto q = ^^float;
    constexpr info v{};

    static_assert(__is_same(decltype(^^int), info));
    static_assert(__is_same(decltype(^^float), info));
    static_assert(__is_same(decltype(^^double), info));
    static_assert(__is_same(decltype(^^long), info));
    static_assert(__is_same(decltype(^^long long), info));
    static_assert(__is_same(decltype(^^short), info));
    static_assert(__is_same(decltype(^^char), info));
    static_assert(__is_same(decltype(^^unsigned char), info));
    static_assert(__is_same(decltype(^^unsigned short), info));
    static_assert(__is_same(decltype(^^unsigned int), info));
    static_assert(__is_same(decltype(^^unsigned long), info));
    static_assert(__is_same(decltype(^^unsigned long long), info));
    static_assert(__is_same(decltype(^^info), info));

    static_assert(__is_same(decltype(^^int), decltype(^^int)));
    static_assert(__is_same(decltype(^^int), decltype(^^float)));
    static_assert(__is_same(decltype(^^int), decltype(^^char)));
    static_assert(__is_same(decltype(^^double), decltype(^^float)));

    static_assert(!__is_same(decltype(^^int), int));
    static_assert(__is_scalar(info));

    static_assert(f1< ^^int >() == ^^int);
    static_assert(f1< ^^A >() != ^^int);
    static_assert(f1< ^^float>() != ^^int);

    static_assert(f2<r>() == ^^int);
    static_assert(f2<^^float>() != ^^int);

    static_assert(sizeof(info) == 8);
    static_assert(alignof(info) == 1);
    static_assert(sizeof(decltype(^^int)) == sizeof(decltype(^^float)));
    static_assert(^^int32_t != ^^int);
    static_assert(^^const int != ^^int);
    static_assert(^^volatile int32_t == ^^volatile int);
    static_assert(^^const volatile int32_t == ^^const volatile int);
    static_assert(^^A != ^^int);


    static_assert(^^int == ^^int);
    static_assert(^^int != ^^float);
    static_assert(^^float != ^^int);
    static_assert(!(^^float == ^^int));
    static_assert(r != q);

    int a;
    static_assert(^^int == ^^decltype(a));

    using foo = const int;
    static_assert(^^foo != ^^const int);
    static_assert(^^const foo == ^^const int);
    static_assert(^^const int == ^^int const);
}
