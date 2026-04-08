// RUN: %clang_cc1 %s -std=c++26 -freflection -fsyntax-only

using info = decltype(^^int);

consteval void test()
{
    constexpr auto r = ^^int;
    constexpr auto q = ^^int;

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

    static_assert(sizeof(^^int) == sizeof(^^float));
    static_assert(sizeof(^^int) == 8);


    static_assert(^^int == ^^int);
    static_assert(^^int != ^^float);
    static_assert(^^float != ^^int);
    static_assert(!(^^float == ^^int));
    static_assert(r == q);

    int a;
    static_assert(^^int == ^^decltype(a));
}
