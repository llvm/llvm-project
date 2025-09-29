// RUN: %clang_cc1 -std=c++20 -triple x86_64-elf-gnu %s -emit-llvm -o - | FileCheck %s

namespace GH56652{

struct foo {};

template <typename T> struct bar {
    using type = T;

    template <foo> inline static constexpr auto b = true;
};

template <typename T>
concept C = requires(T a) { T::template b<foo{}>; };

template <typename T> auto fn(T) {
    if constexpr (!C<T>)
        return foo{};
    else
        return T{};
}

auto a = decltype(fn(bar<int>{})){};

}

// CHECK: %"struct.GH56652::bar" = type { i8 }
// CHECK: @_ZN7GH566521aE = global %"struct.GH56652::bar" undef
