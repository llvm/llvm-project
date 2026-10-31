// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fno-rtti -fsyntax-only -verify -DITANIUM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fno-rtti -fsyntax-only -verify -DITANIUM -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -fno-rtti -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -fno-rtti -fsyntax-only -verify -fexperimental-new-constant-interpreter %s

static_assert(__is_same(decltype(__builtin_mangled_name(int)), const char *));

#ifdef ITANIUM
constexpr char ExpectedIntName[] = "_ZTSi";
#else
constexpr char ExpectedIntName[] = "?AH";
#endif
constexpr const char *IntName = __builtin_mangled_name(int);
static_assert(__builtin_strcmp(IntName, ExpectedIntName) == 0);
static_assert(IntName[sizeof(ExpectedIntName) - 1] == '\0');

using IntAlias = int;
static_assert(__builtin_strcmp(__builtin_mangled_name(IntAlias), IntName) == 0);
static_assert(__builtin_strcmp(__builtin_mangled_name(long), IntName) != 0);
static_assert(__builtin_strcmp(__builtin_mangled_name(const int), IntName) != 0);
static_assert(__builtin_strcmp(__builtin_mangled_name(int &), IntName) != 0);
static_assert(__builtin_strcmp(__builtin_mangled_name(int[3]),
                               __builtin_mangled_name(int[4])) != 0);

struct Incomplete;
static_assert(__builtin_strlen(__builtin_mangled_name(Incomplete)) != 0);
static_assert(__builtin_strlen(__builtin_mangled_name(void)) != 0);
static_assert(__builtin_strlen(__builtin_mangled_name(int())) != 0);

template <class T> constexpr const char *name() {
  return __builtin_mangled_name(T);
}
static_assert(__builtin_strcmp(name<int>(), IntName) == 0);
static_assert(__builtin_strcmp(name<long>(), IntName) != 0);

void invalid(int value) {
  __builtin_mangled_name();          // expected-error {{expected a type}}
  __builtin_mangled_name(int, long); // expected-error {{type trait requires 1 argument; have 2 arguments}}
  __builtin_mangled_name(value);     // expected-error {{unknown type name 'value'}}
}
