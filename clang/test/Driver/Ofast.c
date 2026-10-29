// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -O2 -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -fno-fast-math -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -fno-strict-aliasing -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -fno-vectorize -Ofast -### %s 2>&1 | FileCheck -check-prefixes=NO-VECTORIZE,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Ofast -O2 -### %s 2>&1 | FileCheck -check-prefixes=O2,O2-ALIASING,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Ofast -fno-fast-math -### %s 2>&1 | FileCheck -check-prefixes=NO-FAST-MATH,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Ofast -fno-strict-aliasing -### %s 2>&1 | FileCheck -check-prefixes=NO-STRICT-ALIASING,WARN %s
// RUN: %clang --target=x86_64-unknown-linux-gnu -c -Ofast -fno-vectorize -### %s 2>&1 | FileCheck -check-prefixes=NO-VECTORIZE,WARN %s

// RUN: %clang --target=x86_64-windows-msvc -c -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -O2 -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -fno-fast-math -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -fno-strict-aliasing -Ofast -### %s 2>&1 | FileCheck -check-prefixes=OFAST,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -fno-vectorize -Ofast -### %s 2>&1 | FileCheck -check-prefixes=NO-VECTORIZE,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -Ofast -O2 -### %s 2>&1 | FileCheck -check-prefixes=O2,O2-ALIASING-MSVC,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -Ofast -fno-fast-math -### %s 2>&1 | FileCheck -check-prefixes=NO-FAST-MATH,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -Ofast -fno-strict-aliasing -### %s 2>&1 | FileCheck -check-prefixes=NO-STRICT-ALIASING,WARN %s
// RUN: %clang --target=x86_64-windows-msvc -c -Ofast -fno-vectorize -### %s 2>&1 | FileCheck -check-prefixes=NO-VECTORIZE,WARN %s

// WARN: use '-O3' to enable only conforming optimizations, or consult the documentation for the same behavior

// OFAST: -cc1
// OFAST: -Ofast
// OFAST-NOT: -relaxed-aliasing
// OFAST: -ffast-math
// OFAST: -vectorize-loops

// O2: -cc1
// O2-NOT: -Ofast
// O2-ALIASING-NOT: -relaxed-aliasing
// O2-ALIASING-MSVC: -relaxed-aliasing
// O2-NOT: -ffast-math
// O2: -vectorize-loops

// NO-FAST-MATH: -cc1
// NO-FAST-MATH: -Ofast
// NO-FAST-MATH-NOT: -relaxed-aliasing
// NO-FAST-MATH-NOT: -ffast-math
// NO-FAST-MATH: -vectorize-loops

// NO-STRICT-ALIASING: -cc1
// NO-STRICT-ALIASING: -Ofast
// NO-STRICT-ALIASING: -relaxed-aliasing
// NO-STRICT-ALIASING: -ffast-math
// NO-STRICT-ALIASING: -vectorize-loops

// NO-VECTORIZE: -cc1
// NO-VECTORIZE: -Ofast
// NO-VECTORIZE-NOT: -relaxed-aliasing
// NO-VECTORIZE: -ffast-math
// NO-VECTORIZE-NOT: -vectorize-loops
