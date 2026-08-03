// RUN: %check_clang_tidy -std=c++20-or-later %s readability-redundant-tag %t

struct Struct {};
class Class {};
union Union {};
enum Enum {};

void basic() {
  struct Struct s;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: Struct s;

  class Class c;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'class' keyword in C++ declaration
  // CHECK-FIXES: Class c;

  union Union u;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'union' keyword in C++ declaration
  // CHECK-FIXES: Union u;

  enum Enum e;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'enum' keyword in C++ declaration
  // CHECK-FIXES: Enum e;
}

// Hidden by variable (GitHub issue)
struct Hidden {} Hidden;

void hiddenByVariable() {
  struct Hidden h;
}

// Forward declaration
struct Forward;

void forwardDecl() {
  struct Forward *p;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: Forward *p;
}

// Namespace-qualified type
namespace N {
struct NS {};
}

void namespaceQualified() {
  struct N::NS x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: N::NS x;
}

// Nested type
struct Outer {
  struct Inner {};
};

void nestedType() {
  struct Outer::Inner x;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: Outer::Inner x;
}

// Hidden by function
struct FuncTag {};

void FuncTag();

void hiddenByFunction() {
  struct FuncTag x;
}

// Hidden by enum constant
struct EnumTag {};

enum { EnumTag };

void hiddenByEnumConstant() {
  struct EnumTag x;
}

// Hidden by another variable
struct A {};

A A;

void anotherHiddenVariable() {
  struct A x;
}

// Template argument
template <typename T>
void tf();

void templateArgument() {
  tf<struct Struct>();
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: tf<Struct>();
}

namespace UsingDeclarationRegression {

namespace NS {
struct S {};
} // namespace NS

using NS::S;

using T = struct S;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using T = S;

} // namespace UsingDeclarationRegression

namespace NegativeUsingDeclaration {

namespace NS {
struct S {};
}

using NS::S;

namespace NS1 {

int S;

namespace NS2 {

using T = struct S;

// CHECK-FIXES: using T = struct S;
// CHECK-MESSAGES-NOT: warning:

} // namespace NS2
} // namespace NS1
} // namespace NegativeUsingDeclaration

namespace HiddenNamespaceRegression {

struct S {};

namespace N1 {
namespace N2 {

using A = struct S;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using A = S;

namespace N3 {

int S;

using B = struct S;
// CHECK-FIXES: using B = struct S;

void foo() {
  using C = struct S;
  // CHECK-FIXES: using C = struct S;

  {
    using D = struct S;
    // CHECK-FIXES: using D = struct S;
  }
}

} // namespace N3

void bar() {
  using E = struct S;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: using E = S;
}

} // namespace N2
} // namespace N1

} // namespace HiddenNamespaceRegression

namespace DeepLookupRegression {

struct Global {};

namespace N1 {

using ::DeepLookupRegression::Global;

using T1 = struct Global;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using T1 = Global;

namespace N2 {

using T2 = struct Global;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using T2 = Global;

namespace N3 {

int Global;

using T3 = struct Global;
// CHECK-FIXES: using T3 = struct Global;

namespace N4 {

using T4 = struct Global;
// CHECK-FIXES: using T4 = struct Global;

namespace N5 {

using T5 = struct Global;
// CHECK-FIXES: using T5 = struct Global;

} // namespace N5
} // namespace N4
} // namespace N3

using T6 = struct Global;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using T6 = Global;

} // namespace N2

using T7 = struct Global;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using T7 = Global;

} // namespace N1

} // namespace DeepLookupRegression

namespace DeepBlockRegression {

struct S {};

namespace N1 {

using ::DeepBlockRegression::S;

namespace N2 {

using A = struct S;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using A = S;

namespace N3 {

int S;

using B = struct S;
// CHECK-FIXES: using B = struct S;

void foo() {
  using C = struct S;
  // CHECK-FIXES: using C = struct S;

  {
    using D = struct S;
    // CHECK-FIXES: using D = struct S;

    {
      using E = struct S;
      // CHECK-FIXES: using E = struct S;

      {
        using F = struct S;
        // CHECK-FIXES: using F = struct S;

        {
          using G = struct S;
          // CHECK-FIXES: using G = struct S;
        }
      }
    }
  }
}

} // namespace N3

void bar() {
  using H = struct S;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: using H = S;
}

namespace N4 {

using I = struct S;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: redundant 'struct' keyword in C++ declaration
// CHECK-FIXES: using I = S;

namespace N5 {

void baz() {
  using J = struct S;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: redundant 'struct' keyword in C++ declaration
  // CHECK-FIXES: using J = S;

  {
    using K = struct S;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: redundant 'struct' keyword in C++ declaration
    // CHECK-FIXES: using K = S;
  }
}

} // namespace N5
} // namespace N4
} // namespace N2
} // namespace N1

} // namespace DeepBlockRegression
