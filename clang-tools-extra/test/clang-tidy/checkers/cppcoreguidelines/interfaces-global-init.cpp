// RUN: %check_clang_tidy %s cppcoreguidelines-interfaces-global-init %t

constexpr int makesInt() { return 3; }
constexpr int takesInt(int i) { return i + 1; }
constexpr int takesIntPtr(int *i) { return *i; }

extern int ExternGlobal;
static int GlobalScopeBadInit1 = ExternGlobal;
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'
static int GlobalScopeBadInit2 = takesInt(ExternGlobal);
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'
static int GlobalScopeBadInit3 = takesIntPtr(&ExternGlobal);
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'
static int GlobalScopeBadInit4 = 3 * (ExternGlobal + 2);
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'

namespace ns {
static int NamespaceScope = makesInt();
static int NamespaceScopeBadInit = takesInt(ExternGlobal);
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'

struct A {
  static int ClassScope;
  static int ClassScopeBadInit;
};

int A::ClassScopeBadInit = takesInt(ExternGlobal);
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ExternGlobal'

static int FromClassBadInit = takesInt(A::ClassScope);
// CHECK-MESSAGES: [[@LINE-1]]:12: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'ClassScope'
} // namespace ns

// "const int B::I;" is fine, it just ODR-defines B::I. See [9.4.3] Static
// members [class.static]. However the ODR-definitions are not in the right
// order (C::I after C::J, see [3.6.2.3]).
class B1 {
  static const int I = 0;
  static const int J = I;
};
const int B1::J;
// CHECK-MESSAGES: [[@LINE-1]]:15: warning: initializing non-local variable with non-const expression depending on uninitialized non-local variable 'I'
const int B1::I;

void f() {
  // This is fine, it's executed after dynamic initialization occurs.
  static int G = takesInt(ExternGlobal);
}

// Declaration then definition then usage is fine.
extern int ExternGlobal2;
extern int ExternGlobal2;
int ExternGlobal2 = 123;
static int GlobalScopeGoodInit1 = ExternGlobal2;


// Defined global variables are fine:
static int GlobalScope = makesInt();
static int GlobalScopeGoodInit2 = takesInt(GlobalScope);
static int GlobalScope2 = takesInt(ns::NamespaceScope);
// Enums are fine.
enum Enum { kEnumValue = 1 };
static int GlobalScopeFromEnum = takesInt(kEnumValue);

// Leave constexprs alone.
extern constexpr int GlobalScopeConstexpr = makesInt();
static constexpr int GlobalScopeConstexprOk =
    takesInt(GlobalScopeConstexpr);

// This is a pretty common instance: People are usually not using constexpr, but
// this is what they should write:
static constexpr const char kValue[] = "value";
constexpr int Fingerprint(const char *value) { return 0; }
static int kFingerprint = Fingerprint(kValue);

// This is fine because the ODR-definitions are in the right order (C::J after
// C::I).
class B2 {
  static const int I = 0;
  static const int J = I;
};
const int B2::I;
const int B2::J;

