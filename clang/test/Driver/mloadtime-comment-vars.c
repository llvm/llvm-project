// RUN: %clang -### -target powerpc-ibm-aix -mloadtime-comment-vars=sccsid,version %s 2>&1 | FileCheck %s
// RUN: %clang -### -target powerpc64-ibm-aix -mloadtime-comment-vars=::x,N::x,A::x %s 2>&1 | FileCheck %s --check-prefix=SCOPE
// RUN: %clang -### -target x86_64-linux-gnu -mloadtime-comment-vars=sccsid,version %s 2>&1 | FileCheck %s --check-prefix=NONAIX

// CHECK: "-cc1"
// CHECK-SAME: "-mloadtime-comment-vars=sccsid,version"

// SCOPE: "-cc1"
// SCOPE-SAME: "-mloadtime-comment-vars=::x,N::x,A::x"

// NONAIX: warning: ignoring '-mloadtime-comment-vars=sccsid,version' option as it is not currently supported for target 'x86_64-unknown-linux-gnu'
// NONAIX: "-cc1"
// NONAIX-NOT: "-mloadtime-comment-vars=sccsid,version"

int main(void) { return 0; }
