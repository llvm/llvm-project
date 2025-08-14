// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 -tune-cpu pwr8 %s 2>&1 | FileCheck --check-prefix=P8 %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr9 -tune-cpu pwr8 %s 2>&1 | FileCheck --check-prefix=P8 %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu g5 -tune-cpu pwr8 %s 2>&1 | FileCheck --check-prefix=NONE --allow-empty %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr8 -tune-cpu pwr9 %s 2>&1 | FileCheck --check-prefix=NONE --allow-empty %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr8 -tune-cpu pwr10 %s 2>&1 | FileCheck --check-prefix=NONE --allow-empty %s

// P8: instructions of current target may not be supported by tune CPU 'pwr8'
// NONE-NOT: instructions of current target may not be supported by tune CPU
