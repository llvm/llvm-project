// RUN: %clang -Ofast -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST %s
// RUN: %clang -O2 -Ofast -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST %s
// RUN: %clang -fno-fast-math -Ofast -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST %s
// RUN: %clang -fno-strict-aliasing -Ofast -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST %s
// RUN: %clang -fno-vectorize -Ofast -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST %s
// RUN: %clang -Ofast -O2 -### -Werror %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST-O2 \
// RUN:  %if target={{.*-windows-msvc.*}} %{ --check-prefix=CHECK-OFAST-O2-ALIASING-MSVC %} \
// RUN:  %else %{ --check-prefix=CHECK-OFAST-O2-ALIASING %} %s
// RUN: %clang -Ofast -fno-fast-math -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST-NO-FAST-MATH %s
// RUN: %clang -Ofast -fno-strict-aliasing -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST-NO-STRICT-ALIASING %s
// RUN: %clang -Ofast -fno-vectorize -### %s 2>&1 | FileCheck -check-prefix=CHECK-OFAST-NO-VECTORIZE %s

// CHECK-OFAST: use '-O3 -ffast-math' for the same behavior, or '-O3' to enable only conforming optimizations
// CHECK-OFAST: -cc1
// CHECK-OFAST-NOT: -relaxed-aliasing
// CHECK-OFAST: -ffast-math
// CHECK-OFAST: -Ofast
// CHECK-OFAST: -vectorize-loops

// Lack of warning about '-Ofast' deprecation is checked via -Werror
// CHECK-OFAST-O2: -cc1
// CHECK-OFAST-O2-ALIASING-NOT: -relaxed-aliasing
// CHECK-OFAST-O2-ALIASING-MSVC: -relaxed-aliasing
// CHECK-OFAST-O2-NOT: -ffast-math
// CHECK-OFAST-O2-NOT: -Ofast
// CHECK-OFAST-O2: -vectorize-loops

// CHECK-OFAST-NO-FAST-MATH: use '-O3 -ffast-math' for the same behavior, or '-O3' to enable only conforming optimizations
// CHECK-OFAST-NO-FAST-MATH: -cc1
// CHECK-OFAST-NO-FAST-MATH-NOT: -relaxed-aliasing
// CHECK-OFAST-NO-FAST-MATH-NOT: -ffast-math
// CHECK-OFAST-NO-FAST-MATH: -Ofast
// CHECK-OFAST-NO-FAST-MATH: -vectorize-loops

// CHECK-OFAST-NO-STRICT-ALIASING: use '-O3 -ffast-math' for the same behavior, or '-O3' to enable only conforming optimizations
// CHECK-OFAST-NO-STRICT-ALIASING: -cc1
// CHECK-OFAST-NO-STRICT-ALIASING: -relaxed-aliasing
// CHECK-OFAST-NO-STRICT-ALIASING: -ffast-math
// CHECK-OFAST-NO-STRICT-ALIASING: -Ofast
// CHECK-OFAST-NO-STRICT-ALIASING: -vectorize-loops

// CHECK-OFAST-NO-VECTORIZE: use '-O3 -ffast-math' for the same behavior, or '-O3' to enable only conforming optimizations
// CHECK-OFAST-NO-VECTORIZE: -cc1
// CHECK-OFAST-NO-VECTORIZE-NOT: -relaxed-aliasing
// CHECK-OFAST-NO-VECTORIZE: -ffast-math
// CHECK-OFAST-NO-VECTORIZE: -Ofast
// CHECK-OFAST-NO-VECTORIZE-NOT: -vectorize-loops
