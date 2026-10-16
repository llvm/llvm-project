// RUN: %clang -target x86_64-apple-macosx10.13 -target-variant x86_64-apple-ios13.1-macabi -ftarget-variant-availability-checks -c -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-macosx10.13 -target-variant x86_64-apple-ios13.1-macabi -fno-target-variant-availability-checks -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO_ARG %s
// RUN: %clang -target x86_64-apple-macosx10.13 -target-variant x86_64-apple-ios13.1-macabi -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=NO_ARG %s

// NO_ARG-NOT: -ftarget-variant-availability-checks
// CHECK: -ftarget-variant-availability-checks

