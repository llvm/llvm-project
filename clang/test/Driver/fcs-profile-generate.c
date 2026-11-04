// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/a.profdata %S/Inputs/a.proftext

// RUN: %clang -### -c -fprofile-use=%t/a.profdata -fcs-profile-generate %s 2>&1 | FileCheck %s
// CHECK:      "-fprofile-instrument=csllvm"
// CHECK-NOT:  "-fprofile-instrument-path=
// CHECK-SAME: "-fprofile-instrument-use=llvm"
// CHECK-SAME: "-fprofile-instrument-use-path={{.*}}a.profdata"

// RUN: %clang -### -c -fprofile-use=%t/a.profdata -fcs-profile-generate=dir %s 2>&1 | FileCheck %s --check-prefix=CHECK1
// CHECK1: "-fprofile-instrument=csllvm"{{.*}} "-fprofile-instrument-path=dir{{/|\\\\}}default_%m.profraw" "-fprofile-instrument-use=llvm" "-fprofile-instrument-use-path={{.*}}a.profdata"

/// Degradation case. This usage does not make much sense.
// RUN: %clang -### -c -fcs-profile-generate %s 2>&1 | FileCheck %s --check-prefix=NOUSE
// NOUSE:     "-fprofile-instrument=csllvm"
// NOUSE-NOT: "-fprofile-instrument-path=

// RUN: not %clang -### -c -fprofile-generate -fcs-profile-generate %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// CONFLICT: error: invalid argument '-fcs-profile-generate' not allowed with '-fprofile-generate'
