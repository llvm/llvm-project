// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config crosscheck-with-z3=true -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify %s
//
// REQUIRES: z3
//
// Requires z3 only for refutation. Works with both constraint managers.

void clang_analyzer_dump(int);

using sugar_t = unsigned char;

// Enum types
enum class ScopedSugared : sugar_t {};
enum class ScopedPrimitive : unsigned char {};
enum UnscopedSugared : sugar_t {};
enum UnscopedPrimitive : unsigned char {};

template <typename T>
T conjure();

void test_enum_types() {
  int sym1 = static_cast<unsigned char>(conjure<ScopedSugared>()) & 0x0F;
  int sym2 = static_cast<unsigned char>(conjure<ScopedPrimitive>()) & 0x0F;
  int sym3 = static_cast<unsigned char>(conjure<UnscopedSugared>()) & 0x0F;
  int sym4 = static_cast<unsigned char>(conjure<UnscopedPrimitive>()) & 0x0F;

  if (sym1 && sym2 && sym3 && sym4) {
    // no-crash on these dumps
    clang_analyzer_dump(sym1); // expected-warning{{((unsigned char) (conj_}}
    clang_analyzer_dump(sym2); // expected-warning{{((unsigned char) (conj_}}
    clang_analyzer_dump(sym3); // expected-warning{{(conj_}}
    clang_analyzer_dump(sym4); // expected-warning{{(conj_}}
  }
}
