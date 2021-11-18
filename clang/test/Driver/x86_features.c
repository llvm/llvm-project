// RUN: %clang -target i386-unknown-unknown -### -S %s -msse -msse4 -mno-sse -mno-mmx -msse  2>&1 | FileCheck %s
// CHECK: "pentium4" "-target-feature" "+sse4.2" "-target-feature" "-mmx" "-target-feature" "+sse"
// Note that we filter out all but the last -m(no)sse.

// Test that we don't produce an error with -mieee-fp.
// RUN: %clang -### %s -mieee-fp -S 2>&1 | FileCheck --check-prefix=IEEE %s
// IEEE-NOT: error: unknown argument

// RUN: %clang -target i386-unknown-unknown -### %s -mskip-rax-setup -S 2>&1 | FileCheck --check-prefix=SRS %s
// SRS: "-mskip-rax-setup"

// RUN: %clang -target i386-unknown-unknown -### %s -mno-skip-rax-setup -S 2>&1 | FileCheck --check-prefix=NO-SRS %s
// NO-SRS-NOT: "-mskip-rax-setup"
