// RUN: %clang -### -target powerpc-ibm-aix -mloadtime-comment-vars=sccsid,version %s 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -mloadtime-comment-vars=sccsid,version %s 2>&1 | FileCheck %s --check-prefix=NONAIX

// RUN: %clang -### -target powerpc-ibm-aix -mloadtime-comment-vars=sccsid -mloadtime-comment-vars=version,build %s 2>&1 | FileCheck %s --check-prefix=MULTI
// RUN: %clang -### -target x86_64-linux-gnu -mloadtime-comment-vars=sccsid -mloadtime-comment-vars=version %s 2>&1 | FileCheck %s --check-prefix=NONAIX-MULTI

// Verify the option is forwarded verbatim to cc1 for a supported target.
// CHECK: "-cc1"
// CHECK-SAME: "-mloadtime-comment-vars=sccsid,version"

// Verify a warning is emitted and the option is NOT forwarded for an
// unsupported target.
// NONAIX: warning: ignoring '-mloadtime-comment-vars=' option as it is not currently supported for target 'x86_64-unknown-linux-gnu'
// NONAIX: "-cc1"
// NONAIX-NOT: "-mloadtime-comment-vars=sccsid,version"

// Verify every occurrence is forwarded, in order; cc1 combines the lists.
// MULTI: "-cc1"
// MULTI-SAME: "-mloadtime-comment-vars=sccsid"
// MULTI-SAME: "-mloadtime-comment-vars=version,build"

// Verify a repeated option on an unsupported target warns once and forwards
// nothing.
// NONAIX-MULTI: warning: ignoring '-mloadtime-comment-vars=' option as it is not currently supported for target 'x86_64-unknown-linux-gnu'
// NONAIX-MULTI: "-cc1"
// NONAIX-MULTI-NOT: "-mloadtime-comment-vars=

int main(void) { return 0; }
