// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config crosscheck-with-z3=true -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config support-symbolic-integer-casts=true -verify=symbolic %s
//
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config support-symbolic-integer-casts=false -verify %s
//
// REQUIRES: asserts, z3
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
  // Let's construct a 'binop(sym, int)', where the binop will trigger an
  // integral promotion to int. Note that we need to first explicit cast
  // the scoped-enum to an integral, to make it compile. We could have choosen
  // any other binary operator.
  int sym1 = static_cast<unsigned char>(conjure<ScopedSugared>()) & 0x0F;
  int sym2 = static_cast<unsigned char>(conjure<ScopedPrimitive>()) & 0x0F;
  int sym3 = static_cast<unsigned char>(conjure<UnscopedSugared>()) & 0x0F;
  int sym4 = static_cast<unsigned char>(conjure<UnscopedPrimitive>()) & 0x0F;

  // We need to introduce a constraint referring to the binop, to get it
  // serialized during the z3-refutation.
  if (sym1 && sym2 && sym3 && sym4) {
    // no-crash on these dumps
    //
    // The 'clang_analyzer_dump' will construct a bugreport, which in turn will
    // trigger a z3-refutation. Refutation will walk the bugpath, collect and
    // serialize the path-constraints into z3 expressions. The binop will
    // operate over 'int' type, but the symbolic operand might have a different
    // type - surprisingly.
    // Historically, the static analyzer did not emit symbolic casts in a lot
    // of cases, not even when the c++ standard mandated it, like for casting
    // a scoped enum to its underlying type. Hence, during the z3 constraint
    // serialization, it made up these 'missing' integral casts - for the
    // implicit cases at least.
    // However, it would still not emit the cast for missing explicit casts,
    // hence 8-bit wide symbol would be bitwise 'and'-ed with a 32-bit wide
    // int, violating an assertion stating that the operands should have the
    // same bitwidths.

    clang_analyzer_dump(sym1);
    // expected-warning-re@-1 {{((unsigned char) (conj_${{[0-9]+}}{enum ScopedSugared, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}
    // symbolic-warning-re@-2           {{((int) (conj_${{[0-9]+}}{enum ScopedSugared, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}

    clang_analyzer_dump(sym2);
    // expected-warning-re@-1 {{((unsigned char) (conj_${{[0-9]+}}{enum ScopedPrimitive, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}
    // symbolic-warning-re@-2           {{((int) (conj_${{[0-9]+}}{enum ScopedPrimitive, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}

    clang_analyzer_dump(sym3);
    // expected-warning-re@-1        {{(conj_${{[0-9]+}}{enum UnscopedSugared, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}}) & 15}}
    // symbolic-warning-re@-2 {{((int) (conj_${{[0-9]+}}{enum UnscopedSugared, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}

    clang_analyzer_dump(sym4);
    // expected-warning-re@-1        {{(conj_${{[0-9]+}}{enum UnscopedPrimitive, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}}) & 15}}
    // symbolic-warning-re@-2 {{((int) (conj_${{[0-9]+}}{enum UnscopedPrimitive, LC{{[0-9]+}}, S{{[0-9]+}}, #{{[0-9]+}}})) & 15}}
  }
}

