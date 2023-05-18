// RUN: not %clang -Xclang -fexperimental-strict-floating-point \
// RUN: -Xclang -triple -Xclang x86_64-linux-gnu -fapprox-func \
// RUN: -Xclang -verify -ffp-eval-method=source %s 2>&1  \
// RUN: | FileCheck %s --check-prefixes=CHECK-FUNC

// RUN: not %clang -Xclang -fexperimental-strict-floating-point \
// RUN: -Xclang -triple -Xclang x86_64-linux-gnu -Xclang -mreassociate \
// RUN: -ffp-eval-method=source -Xclang -verify %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-ASSOC

// RUN: not %clang -Xclang -fexperimental-strict-floating-point \
// RUN: -Xclang -triple -Xclang x86_64-linux-gnu -Xclang -freciprocal-math \
// RUN: -ffp-eval-method=source -Xclang -verify %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-RECPR

// RUN: not %clang -Xclang -fexperimental-strict-floating-point \
// RUN: -Xclang -triple -Xclang x86_64-linux-gnu -Xclang -freciprocal-math \
// RUN: -Xclang -mreassociate -ffp-eval-method=source -Xclang -verify %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-ASSOC,CHECK-RECPR

// RUN: not %clang -Xclang -fexperimental-strict-floating-point \
// RUN: -Xclang -triple -Xclang x86_64-linux-gnu -Xclang -freciprocal-math \
// RUN: -Xclang -mreassociate -fapprox-func -ffp-eval-method=source \
// RUN: -Xclang -verify %s 2>&1 \
// RUN: | FileCheck %s --check-prefixes=CHECK-ASSOC,CHECK-RECPR,CHECK-FUNC

// CHECK-FUNC: (frontend): option 'ffp-eval-method' cannot be used with option 'fapprox-func'
// CHECK-ASSOC: (frontend): option 'ffp-eval-method' cannot be used with option 'mreassociate'
// CHECK-RECPR: (frontend): option 'ffp-eval-method' cannot be used with option 'freciprocal'
