// Tests for static analyzer checkers that the driver enables by default based
// on the target triple.

// RUN: %clang -### -target x86_64-apple-darwin10 --analyze %s 2>&1 | FileCheck --check-prefix=CHECK-DARWIN %s

// CHECK-DARWIN: "-analyzer-checker=core"
// CHECK-DARWIN-SAME: "-analyzer-checker=apiModeling"
// CHECK-DARWIN-SAME: "-analyzer-checker=unix"
// CHECK-DARWIN-SAME: "-analyzer-checker=osx"
// CHECK-DARWIN-SAME: "-analyzer-checker=deadcode"
// CHECK-DARWIN-SAME: "-analyzer-checker=cplusplus"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.UncheckedReturn"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.getpw"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.gets"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.mktemp"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.mkstemp"
// CHECK-DARWIN-SAME: "-analyzer-checker=security.insecureAPI.vfork"
// CHECK-DARWIN-SAME: "-analyzer-checker=nullability.NullPassedToNonnull"
// CHECK-DARWIN-SAME: "-analyzer-checker=nullability.NullReturnedFromNonnull"


// RUN: %clang -### -target x86_64-unknown-linux --analyze %s 2>&1 | FileCheck --check-prefix=CHECK-LINUX %s

// CHECK-LINUX: "-analyzer-checker=core"
// CHECK-LINUX-SAME: "-analyzer-checker=apiModeling"
// CHECK-LINUX-SAME: "-analyzer-checker=unix"
// CHECK-LINUX-NOT:  "-analyzer-checker=osx"
// CHECK-LINUX-SAME: "-analyzer-checker=deadcode"
// CHECK-LINUX-SAME: "-analyzer-checker=cplusplus"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.UncheckedReturn"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.getpw"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.gets"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.mktemp"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.mkstemp"
// CHECK-LINUX-SAME: "-analyzer-checker=security.insecureAPI.vfork"
// CHECK-LINUX-SAME: "-analyzer-checker=nullability.NullPassedToNonnull"
// CHECK-LINUX-SAME: "-analyzer-checker=nullability.NullReturnedFromNonnull"


// RUN: %clang -### -target x86_64-windows --analyze %s 2>&1 | FileCheck --check-prefix=CHECK-WINDOWS %s

// CHECK-WINDOWS: "-analyzer-checker=core"
// CHECK-WINDOWS-SAME: "-analyzer-checker=apiModeling"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.API"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.Malloc"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.MallocSizeof"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.MismatchedDeallocator"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.cstring.BadSizeArg"
// CHECK-WINDOWS-SAME: "-analyzer-checker=unix.cstring.NullArg"
// CHECK-WINDOWS-NOT:  "-analyzer-checker=osx"
// CHECK-WINDOWS-SAME: "-analyzer-checker=deadcode"
// CHECK-WINDOWS-SAME: "-analyzer-checker=cplusplus"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.UncheckedReturn"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.getpw"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.gets"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.mktemp"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.mkstemp"
// CHECK-WINDOWS-SAME: "-analyzer-checker=security.insecureAPI.vfork"
// CHECK-WINDOWS-SAME: "-analyzer-checker=nullability.NullPassedToNonnull"
// CHECK-WINDOWS-SAME: "-analyzer-checker=nullability.NullReturnedFromNonnull"
