// RUN: %check_clang_tidy -std=c++17-or-later \
// RUN:   -check-header %S/Inputs/macro-to-enum/namespace-values.h \
// RUN:   %s modernize-macro-to-enum %t -- \
// RUN:   -- -I%S/Inputs/macro-to-enum -fno-delayed-template-parsing
// RUN: %clang -std=c++17 -fsyntax-only -I%t.cpp.headers %t.cpp

// Translation-unit definitions remain usable from any namespace.
#define GLOBAL_VALUE 10
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'GLOBAL_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: GLOBAL_VALUE = 10
// CHECK-FIXES-NEXT: };

namespace Same {

#define SAME_VALUE 11
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'SAME_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: SAME_VALUE = 11
// CHECK-FIXES-NEXT: };

int First = SAME_VALUE + GLOBAL_VALUE;
int function();
struct Object {
  int method();
};
}

// Reopened blocks and out-of-line definitions retain their namespace.
namespace Same {
int Second = SAME_VALUE;
}

int Same::function() { return SAME_VALUE; }
int Same::Object::method() { return SAME_VALUE; }

namespace Same {
int parameter(int Value);
auto trailing() -> decltype(SAME_VALUE);
}

int Same::parameter(int Value = SAME_VALUE) { return Value; }
auto Same::trailing() -> decltype(SAME_VALUE) { return SAME_VALUE; }

// A leading return type is outside the qualified function's namespace.
namespace ReturnTypes {
#define RETURN_SIZE 23
// CHECK-FIXES: #define RETURN_SIZE 23

template <int N> struct Buffer {};
Buffer<RETURN_SIZE> make();
}

ReturnTypes::Buffer<RETURN_SIZE> ReturnTypes::make() { return {}; }

// Specialization arguments precede the class scope, even after its name.
namespace Specializations {
#define SPECIALIZATION_SIZE 24
// CHECK-FIXES: #define SPECIALIZATION_SIZE 24

template <int N> struct Buffer;
}

template <> struct Specializations::Buffer<SPECIALIZATION_SIZE> {};

namespace Specializations {
#define FUNCTION_ARGUMENT_SIZE 26
// CHECK-FIXES: #define FUNCTION_ARGUMENT_SIZE 26

template <int N> int function();
}

template <> int Specializations::function<FUNCTION_ARGUMENT_SIZE>() { return 0; }

namespace Initializers {
#define INITIALIZER_VALUE 25
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'INITIALIZER_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: INITIALIZER_VALUE = 25
// CHECK-FIXES-NEXT: };

extern int Value;
}

int Initializers::Value = INITIALIZER_VALUE;

namespace Outer {

#define OUTER_VALUE 12
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'OUTER_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: OUTER_VALUE = 12
// CHECK-FIXES-NEXT: };

namespace Child {
int NestedUse = OUTER_VALUE;
}
}

// Qualified namespace syntax can give nested declarations shared ranges.
namespace Outer::Inner {

#define INNER_VALUE 13
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'INNER_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: INNER_VALUE = 13
// CHECK-FIXES-NEXT: };

int Value = INNER_VALUE;
int DescendantUse = OUTER_VALUE;

#define INNER_ESCAPE 14
// CHECK-FIXES: #define INNER_ESCAPE 14

int SafeUse = INNER_ESCAPE;
}

namespace Outer::Inner {
int ReopenedUse = INNER_VALUE;
}

namespace Outer {
// A use in the parent is outside the defining inner namespace.
int ParentUse = INNER_ESCAPE;
}

// One outside use rejects the whole cluster, including its safely used member.
namespace Mixed {
#define GROUP_SAFE 15
#define GROUP_ESCAPE 16
// CHECK-FIXES: #define GROUP_SAFE 15
// CHECK-FIXES-NEXT: #define GROUP_ESCAPE 16

int Local = GROUP_SAFE;
}

int Outside = GROUP_ESCAPE;

namespace Mixed {
#define MIXED_USES 17
// CHECK-FIXES: #define MIXED_USES 17

int LocalMixed = MIXED_USES;
}

namespace Other {
int OtherMixed = MIXED_USES;
}

// Object-like aliases are checked at the eventual use, not their definition.
namespace Aliases {
#define ALIAS_VALUE 18
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'ALIAS_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: ALIAS_VALUE = 18
// CHECK-FIXES-NEXT: };
}

#define VALUE_ALIAS ALIAS_VALUE

namespace Aliases {
int Value = VALUE_ALIAS;
}

// Separate anonymous namespace blocks refer to the same namespace.
namespace {
#define ANONYMOUS_VALUE 19
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'ANONYMOUS_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: ANONYMOUS_VALUE = 19
// CHECK-FIXES-NEXT: };

int AnonymousFirst = ANONYMOUS_VALUE;
}

namespace {
int AnonymousSecond = ANONYMOUS_VALUE;
}

namespace Templates {
#define TEMPLATE_VALUE 20
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: replace macro with enum
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: macro 'TEMPLATE_VALUE' defines an integral constant; prefer an enum instead
// CHECK-FIXES: enum {
// CHECK-FIXES-NEXT: TEMPLATE_VALUE = 20
// CHECK-FIXES-NEXT: };

template <typename T> int value(T) { return TEMPLATE_VALUE; }
}

int FirstInstance = Templates::value(0);
int SecondInstance = Templates::value(0L);

// Included definitions inherit the namespace, through linkage blocks as well.
namespace Headers {
extern "C" {
#include "namespace-values.h"
}
}

namespace Headers {
#include "namespace-values.h"
int HeaderUse = NAMESPACE_HEADER_VALUE;
}

int HeaderEscape = NAMESPACE_HEADER_ESCAPE;

// Namespace support must not admit definitions inside functions or classes.
namespace LocalScopes {
int function() {
#define NAMESPACE_FUNCTION_LOCAL 21
  // CHECK-FIXES: #define NAMESPACE_FUNCTION_LOCAL 21
  return NAMESPACE_FUNCTION_LOCAL;
}

int AfterFunction = NAMESPACE_FUNCTION_LOCAL;

struct Object {
#define NAMESPACE_CLASS_LOCAL 22
  // CHECK-FIXES: #define NAMESPACE_CLASS_LOCAL 22
  int Value;
};

int AfterClass = NAMESPACE_CLASS_LOCAL;
}
