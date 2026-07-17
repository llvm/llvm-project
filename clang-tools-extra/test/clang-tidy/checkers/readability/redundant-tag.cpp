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
