// RUN: %clang_analyze_cc1 -triple=x86_64-unknown-linux -analyzer-checker=core,debug.ExprInspection -analyzer-config eagerly-assume=false -verify %s

void clang_analyzer_dump(int);
void clang_analyzer_dumpL(long);
void clang_analyzer_warnIfReached();

void testInspect(int x) {
  if ((x < 10) || (x > 100)) {
    return;
  }
  // x: [10, 100]

  int i = x + 1;
  long l = i - 10U;
  clang_analyzer_dump(i);       // expected-warning-re {{(reg_${{[0-9]+}}<int x>) + 1 }}
  clang_analyzer_dumpL(l);      // expected-warning-re {{(reg_${{[0-9]+}}<int x>) - 9U }} instead of + 4294967287U
  clang_analyzer_dumpL(l + 0L); // expected-warning-re {{(reg_${{[0-9]+}}<int x>) - 9 }}  instead of + 4294967287

  if ((l - 1000) > 0) {
    clang_analyzer_warnIfReached(); // no-warning
  }
  if (l > 1000) {
    clang_analyzer_warnIfReached(); // no-warning
  }
  if (l > 1000L) {
    clang_analyzer_warnIfReached(); // no-warning
  }
  if ((l + 0L) > 1000) {
    clang_analyzer_warnIfReached(); // no-warning
  }

  i = x - 1;
  l = i + 10U;
  clang_analyzer_dumpL(l); // expected-warning-re {{(reg_${{[0-9]+}}<int x>) + 9U }} instead of - 4294967287U

  i = x + (-1);
  l = i - 10U;
  clang_analyzer_dumpL(l); // expected-warning-re {{(reg_${{[0-9]+}}<int x>) - 11U }} instead of + 4294967285U

  i = x + 1U;
  l = i - 10U;
  clang_analyzer_dumpL(l); // expected-warning-re {{(reg_${{[0-9]+}}<int x>) - 9U }} instead of + 4294967287U
}

void testMin(int i, long l) {
  clang_analyzer_dump(i + (-1));  // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - 1 }} instead of + -1
  clang_analyzer_dump(i - (-1));  // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + 1 }} instead of - -1
  clang_analyzer_dumpL(l + (-1)); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 1 }} instead of + -1
  clang_analyzer_dumpL(l - (-1)); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 1 }} instead of - -1

  int intMin = 1 << (sizeof(int) * 8 - 1); // INT_MIN, negative value is not representable
  // Do not normalize representation if negation would not be representable
  clang_analyzer_dump(i + intMin); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + -2147483648 }}
  clang_analyzer_dump(i - intMin); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - -2147483648 }}
  // Produced value has higher bit with (long) so negation if representable
  clang_analyzer_dumpL(l + intMin); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 2147483648 }} instead of + -2147483648
  clang_analyzer_dumpL(l - intMin); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 2147483648 }} instead of - -2147483648
}
