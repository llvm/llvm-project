// RUN: rm -rf %t
// RUN: mkdir %t

// Check that -fdriver-only doesn't actually run the generated -cc1 job.
//
// RUN: %clang -c %s -o %t/a.o -fdriver-only
// RUN: not cat %t/a.o

// Check that -fdriver-only respects errors.
//
// RUN: not %clang -c %s -fdriver-only -target i386-apple-darwin9 -m32 -Xarch_i386 -o

// Check that -fdriver-only respects -v.
//
// RUN: %clang -c %s -fdriver-only -v 2>&1 | FileCheck %s --check-prefix=CHECK-V
// CHECK-V: {{.*}} "-cc1"
//
// RUN: %clang -c %s -fdriver-only    2>&1 | FileCheck %s --check-prefix=CHECK-NO-V --allow-empty
// CHECK-NO-V-NOT: {{.*}} "-cc1"
