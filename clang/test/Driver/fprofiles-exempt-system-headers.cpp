// The system-header exemption is on by default, so the driver forwards
// -fno-profiles-exempt-system-headers to -cc1 only when the user disables it.
// (Patterns match the -fno- spelling specifically; the exemption's cc1 default
// is on, so nothing is emitted otherwise. The bare option name is avoided in
// CHECKs because it also appears in this file's own name in the -### output.)

// RUN: %clang -### -fprofiles -std=c++23 -c %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -fprofiles -fprofiles-exempt-system-headers -std=c++23 -c %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// RUN: %clang -### -fprofiles -fno-profiles-exempt-system-headers -std=c++23 -c %s 2>&1 | FileCheck -check-prefix=OFF %s

// DEFAULT-NOT: "-fno-profiles-exempt-system-headers"
// OFF: "-fno-profiles-exempt-system-headers"
