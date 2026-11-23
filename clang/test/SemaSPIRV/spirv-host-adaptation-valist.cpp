/// Tests that getBuiltinVaListKind() delegates to the host target.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple aarch64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s

// expected-no-diagnostics

template<typename T, typename U>
struct same_type;
template<typename T>
struct same_type<T, T> {
  using type = int;
};
template<typename T, typename U, typename = typename same_type<T, U>::type>
constexpr bool is_same_type(int) { return true; }
template<typename T, typename U>
constexpr bool is_same_type(...) { return false; }

#if defined(_WIN32)
static_assert(is_same_type<__builtin_va_list, char*>(0));
#else
static_assert(!is_same_type<__builtin_va_list, char*>(0));
#endif
