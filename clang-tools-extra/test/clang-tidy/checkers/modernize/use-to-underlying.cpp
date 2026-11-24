// RUN: %check_clang_tidy -std=c++23-or-later %s modernize-use-to-underlying %t

// The <utility> header for std::to_underlying is not yet included, so the check
// must insert it.
// CHECK-FIXES: #include <utility>

enum class ColorInt : int { Red, Green, Blue };
enum class ByteEnum : unsigned char { A, B };
enum class DefaultEnum { X, Y }; // underlying type is int
enum class OtherEnum : int {};
enum Unscoped : int { U0, U1 }; // not a scoped enum

// A precise cast (destination type equals the underlying type) is always
// diagnosed and the whole cast is replaced. static_cast, C-style and
// functional-style casts are all matched.
void precise(ColorInt c, ByteEnum b, DefaultEnum d) {
  int A = static_cast<int>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int A = std::to_underlying(c);
  unsigned char B = static_cast<unsigned char>(b);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: unsigned char B = std::to_underlying(b);
  int C = static_cast<int>(d);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int C = std::to_underlying(d);
  int D = (int)c;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int D = std::to_underlying(c);
  int E = int(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int E = std::to_underlying(c);
}

// With the default ImpreciseCasts=Warn, an imprecise cast (destination type
// differs from the underlying type in width or signedness) is diagnosed but no
// fix-it is applied.
void imprecise(ColorInt c) {
  long W = static_cast<long>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: long W = static_cast<long>(c);
  unsigned S = static_cast<unsigned>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: unsigned S = static_cast<unsigned>(c);
}

void negatives(ColorInt c, Unscoped u) {
  bool Truthy = static_cast<bool>(c);      // truthiness test, not underlying
  double Dbl = static_cast<double>(c);     // non-integer destination
  OtherEnum Enum = static_cast<OtherEnum>(c); // enum-to-enum
  int Un = static_cast<int>(u);            // unscoped enumeration
}

// When bool is the exact underlying type, the cast is precise and flagged.
enum class BoolEnum : bool { No, Yes };
bool precise_bool(BoolEnum b) {
  return static_cast<bool>(b);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(b);
}

// The operand can be an arbitrary expression, which is preserved verbatim.
int operand_expression(ColorInt a, bool cond) {
  return static_cast<int>(cond ? a : ColorInt::Red);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(cond ? a : ColorInt::Red);
}

// A typedef of a scoped enum is seen through (canonical type is the enum).
using ColorAlias = ColorInt;
int typedef_enum(ColorAlias c) {
  return static_cast<int>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(c);
}

// A cast whose operand has a dependent (template parameter) type is not a
// scoped enumeration in the template pattern, so it is never flagged.
template <typename T>
int dependent_operand(T t) {
  return static_cast<int>(t);
}

// A cast whose destination type is dependent is not a known integer type, so it
// is never flagged either.
template <typename T>
T dependent_destination(ColorInt c) {
  return static_cast<T>(c);
}

// A non-dependent cast inside a template is flagged and fixed on the template
// pattern itself, even when the template is never instantiated and the cast
// sits in a value-dependent context.
template <int N>
int nondependent_in_uninstantiated_template(ColorInt c) {
  return static_cast<int>(c) + N;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(c) + N;
}

// A non-dependent cast that is instantiated multiple times is still flagged and
// fixed exactly once (on the pattern), not once per instantiation.
template <typename T>
int nondependent_in_instantiated_template(ColorInt c) {
  return static_cast<int>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: return std::to_underlying(c);
}

void use_templates(ColorInt c) {
  dependent_operand(c);
  dependent_destination<int>(c);
  dependent_destination<long>(c);
  nondependent_in_instantiated_template<int>(c);
  nondependent_in_instantiated_template<char>(c);
}

// Casts involving macros. The fix is applied whenever the operand's spelling
// can be recovered: at the invocation of a function-like macro whose body is
// the whole cast, when only the destination type comes from a macro, and when
// the operand itself is a macro.
#define CAST_TO_INT(x) static_cast<int>(x)
#define DEST_TYPE int
#define RED_VALUE ColorInt::Red
void macros(ColorInt c) {
  int A = CAST_TO_INT(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int A = std::to_underlying(c);
  int B = static_cast<DEST_TYPE>(c);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int B = std::to_underlying(c);
  int C = static_cast<int>(RED_VALUE);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use 'std::to_underlying' to convert a scoped enumeration to its underlying type [modernize-use-to-underlying]
  // CHECK-FIXES: int C = std::to_underlying(RED_VALUE);
}
