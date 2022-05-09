// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -Wno-strict-prototypes -fsyntax-only -verify -std=c2x %s

struct S {};
struct S * [[clang::address_space(1)]] Foo;

enum [[clang::enum_extensibility(open)]] EnumOpen {
  C0 = 1, C1 = 10
};

enum [[clang::flag_enum]] EnumFlag {
  D0 = 1, D1 = 8
};

[[clang::overloadable]] void foo(void *c);
[[clang::overloadable]] void foo(char *c);

void context_okay(void *context [[clang::swift_context]]) [[clang::swiftcall]];
void context_okay2(void *context [[clang::swift_context]], void *selfType, char **selfWitnessTable) [[clang::swiftcall]];
void context_async_okay(void *context [[clang::swift_async_context]]) [[clang::swiftasynccall]];
void context_async_okay2(void *context [[clang::swift_async_context]], void *selfType, char **selfWitnessTable) [[clang::swiftasynccall]];

[[clang::ownership_returns(foo)]] void *f1(void);
[[clang::ownership_returns(foo)]] void *f2();

[[clang::unavailable("not available - replaced")]] void foo2(void); // expected-note {{'foo2' has been explicitly marked unavailable here}}
void bar(void) {
  foo2(); // expected-error {{'foo2' is unavailable: not available - replaced}}
}

[[nodiscard]] int without_underscores(void);
[[__nodiscard__]] int underscores(void);

// Match GCC's behavior for C attributes as well.
[[gnu::constructor]] void ctor_func(void);
[[gnu::destructor]] void dtor_func(void);
[[gnu::hot]] void hot_func(void);
[[__gnu__::hot]] void hot_func2(void);
[[gnu::__hot__]] void hot_func3(void);
[[__gnu__::__hot__]] void hot_func4(void);

// Note how not all GCC attributes are supported in C.
[[gnu::abi_tag("")]] void abi_func(void); // expected-warning {{unknown attribute 'abi_tag' ignored}}
struct S s [[gnu::init_priority(1)]]; // expected-warning {{unknown attribute 'init_priority' ignored}}
