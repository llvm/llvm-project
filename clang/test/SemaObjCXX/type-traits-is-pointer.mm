// RUN: %clang_cc1 -fsyntax-only -fobjc-arc -fobjc-runtime-has-weak -verify %s
// expected-no-diagnostics

template <typename T>
void test_is_pointer() {
    static_assert(__is_pointer(T), "");

    static_assert(__is_pointer(T __weak), "");
    static_assert(__is_pointer(T __strong), "");
    static_assert(__is_pointer(T __autoreleasing), "");
    static_assert(__is_pointer(T __unsafe_unretained), "");

    static_assert(__is_pointer(T __weak const), "");
    static_assert(__is_pointer(T __strong const), "");
    static_assert(__is_pointer(T __autoreleasing const), "");
    static_assert(__is_pointer(T __unsafe_unretained const), "");

    static_assert(__is_pointer(T __weak volatile), "");
    static_assert(__is_pointer(T __strong volatile), "");
    static_assert(__is_pointer(T __autoreleasing volatile), "");
    static_assert(__is_pointer(T __unsafe_unretained volatile), "");

    static_assert(__is_pointer(T __weak const volatile), "");
    static_assert(__is_pointer(T __strong const volatile), "");
    static_assert(__is_pointer(T __autoreleasing const volatile), "");
    static_assert(__is_pointer(T __unsafe_unretained const volatile), "");
}

@class Foo;

int main(int, char**) {
    test_is_pointer<id>();
    test_is_pointer<id const>();
    test_is_pointer<id volatile>();
    test_is_pointer<id const volatile>();

    test_is_pointer<Foo*>();
    test_is_pointer<Foo const*>();
    test_is_pointer<Foo volatile*>();
    test_is_pointer<Foo const volatile*>();

    test_is_pointer<void*>();
    test_is_pointer<void const*>();
    test_is_pointer<void volatile*>();
    test_is_pointer<void const volatile*>();

    return 0;
}
