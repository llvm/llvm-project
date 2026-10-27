// Nullable evidence needs the value to be null on every path to the use. A
// pointer null on some path only (here, one that returns on an error code
// first, which the analysis cannot see) is maybe-null evidence: it does not
// make the parameter _Nullable, but it still rules out _Nonnull. So is a
// ternary with a null arm.

// RUN: rm -f %t.json
// RUN: %clang_cc1 -fsyntax-only -fnullability-default=nonnull %s \
// RUN:   --ssaf-extract-summaries=NullabilitySafety \
// RUN:   --ssaf-compilation-unit-id=tu --ssaf-tu-summary-file=%t.json
// RUN: %python %S/Inputs/decode-summary.py %t.json | FileCheck %s

int open_it(int **out);
void error_path(int *p);
void every_path(int *p);
void straight(int *p);
void ternary_arm(const char *fmt, const char *arg);

int correlated(int ok) {
  int *p = 0;
  int rc = 1;
  if (ok)
    rc = open_it(&p);
  if (rc)
    return rc;
  error_path(p);
  return 0;
}

void null_on_both(int c) {
  int *p = 0;
  if (c)
    p = 0;
  every_path(p);
}

void straight_line(void) {
  int *p = 0;
  straight(p);
}

void null_arm(const char *e) {
  ternary_arm(e ? "%s" : (char *)0, e);
}

// CHECK-NOT:  {{.}}
// CHECK:      c:@F@correlated MaybeNullEvidence c:@F@error_path param 1
// CHECK-NEXT: c:@F@correlated NonnullEvidence c:@F@open_it param 1
// CHECK-NEXT: c:@F@null_arm ConditionalEvidence c:@F@ternary_arm param 2 <- c:@F@null_arm param 1
// CHECK-NEXT: c:@F@null_arm MaybeNullEvidence c:@F@null_arm param 1
// CHECK-NEXT: c:@F@null_arm MaybeNullEvidence c:@F@ternary_arm param 1
// CHECK-NEXT: c:@F@null_on_both NullableEvidence c:@F@every_path param 1
// CHECK-NEXT: c:@F@straight_line NullableEvidence c:@F@straight param 1
// CHECK-NOT:  {{.}}
