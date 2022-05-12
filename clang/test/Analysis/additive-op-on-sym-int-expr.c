// RUN: %clang_analyze_cc1 -triple=x86_64-unknown-linux-gnu -analyzer-checker=core,apiModeling,debug.ExprInspection -analyzer-config eagerly-assume=false -verify %s

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

const int intMin = 1 << (sizeof(int) * 8 - 1);     // INT_MIN, negation value is not representable
const long longMin = 1L << (sizeof(long) * 8 - 1); // LONG_MIN, negation value is not representable

void testMin(int i, long l) {
  clang_analyzer_dump(i + (-1));  // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - 1 }} instead of + -1
  clang_analyzer_dump(i - (-1));  // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + 1 }} instead of - -1
  clang_analyzer_dumpL(l + (-1)); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 1 }} instead of + -1
  clang_analyzer_dumpL(l - (-1)); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 1 }} instead of - -1

  // Do not normalize representation if negation would not be representable
  clang_analyzer_dump(i + intMin); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + -2147483648 }} no change
  clang_analyzer_dump(i - intMin); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - -2147483648 }} no change
  // Produced value has higher bit with (long) so negation if representable
  clang_analyzer_dumpL(l + intMin); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 2147483648 }} instead of + -2147483648
  clang_analyzer_dumpL(l - intMin); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 2147483648 }} instead of - -2147483648
}

void changingToUnsinged(unsigned u, int i) {
  unsigned c = u + (unsigned)i;
  unsigned d = u - (unsigned)i;
  if (i == -1) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) - 1U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) + 1U }}
    return;
  }
  if (i == intMin) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) - 2147483648U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) + 2147483648U }}
    return;
  }
}

void extendingToSigned(long l, int i) {
  long c = l + (long)i;
  long d = l - (long)i;
  if (i == -1) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 1 }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 1 }}
    return;
  }
  if (i == intMin) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) - 2147483648 }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<long l>) + 2147483648 }}
    return;
  }
}

void extendingToUnigned(unsigned long ul, int i) {
  unsigned long c = ul + (unsigned long)i;
  unsigned long d = ul - (unsigned long)i;
  if (i == -1) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned long ul>) - 1U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned long ul>) + 1U }}
    return;
  }
  if (i == intMin) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned long ul>) - 2147483648U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned long ul>) + 2147483648U }}
    return;
  }
}

void truncatingToSigned(int i, long l) {
  int c = i + (int)l;
  int d = i - (int)l;
  if (l == -1L) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - 1 }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + 1 }}
    return;
  }
  if (l == (long)intMin) {      // negation outside of range, no-changes
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + -2147483648 }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + -2147483648 }}
    return;
  }
  if (l == ((long)intMin - 1L)) { // outside or range, no changes
    clang_analyzer_dump(c + 0);   // expected-warning-re {{(reg_${{[0-9]+}}<int i>) + 2147483647 }}
    clang_analyzer_dump(d + 0);   // expected-warning-re {{(reg_${{[0-9]+}}<int i>) - 2147483647 }}
    return;
  }
  if (l == longMin) {           // outside of range, no changes
    clang_analyzer_dump(c + 0); // expected-warning-re {{reg_${{[0-9]+}}<int i> }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{reg_${{[0-9]+}}<int i> }}
    return;
  }
}

void truncatingToUnsigned(unsigned u, long l) {
  unsigned c = u + (unsigned)l;
  unsigned d = u - (unsigned)l;
  if (l == -1L) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) - 1U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) + 1U }}
    return;
  }
  if (l == (long)intMin) {
    clang_analyzer_dump(c + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) - 2147483648U }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) + 2147483648U }}
    return;
  }
  if (l == ((long)intMin - 1L)) { // outside or range, no changes
    clang_analyzer_dump(c + 0);   // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) + 2147483647U }}
    clang_analyzer_dump(d + 0);   // expected-warning-re {{(reg_${{[0-9]+}}<unsigned int u>) - 2147483647U }}
    return;
  }
  if (l == longMin) {           // outside of range, no changes
    clang_analyzer_dump(c + 0); // expected-warning-re {{reg_${{[0-9]+}}<unsigned int u> }}
    clang_analyzer_dump(d + 0); // expected-warning-re {{reg_${{[0-9]+}}<unsigned int u> }}
    return;
  }
}

// Test for crashes
typedef long ssize_t;
ssize_t write(int, const void *, unsigned long);

int crashTest(int x, int fd) {
  unsigned wres = write(fd, "a", 1);
  if (wres) {
  }
  int t1 = x - wres;
  if (wres < 0) {
  }
  return x + t1; // no crash
}
