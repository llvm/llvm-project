// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fcx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -fcx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN2 %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -fcx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -fno-cx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN4 %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   -fno-cx-fortran-rules | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN3 %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck  %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fcx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN1 %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=WARN4 %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules \
// RUN: -fcx-limited-range  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN20 %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: not %clang -### -target x86_64 -fcomplex-arithmetic=foo -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=ERR %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=PRMTD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcx-limited-range \
// RUN: -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN6 %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules \
// RUN: -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN7 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic \
// RUN: -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=PRMTD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=improved \
// RUN: -fcomplex-arithmetic=promoted  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=PRMTD %s


// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcx-limited-range  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN14 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=improved  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -ffast-math  -c %s 2>&1 | FileCheck --check-prefix=WARN17 %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=improved  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=promoted  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=PRMTD %s

// RUN: %clang -### -target x86_64 -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fno-cx-limited-range -c %s \
// RUN:   2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -ffast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=basic -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -Werror -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -ffast-math -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -ffast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### --target=x86_64 -fcx-limited-range -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE,WARN21 %s

// RUN: %clang -### -Werror --target=x86_64 -fno-cx-limited-range -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcx-fortran-rules -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE,WARN22 %s

// RUN: %clang -### -Werror --target=x86_64 -fno-cx-fortran-rules -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffast-math -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=basic -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE,WARN23 %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=promoted -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE,WARN24 %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=improved -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE,WARN25 %s

// RUN: %clang -### -Werror --target=x86_64 -fcomplex-arithmetic=full -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffp-model=aggressive -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffp-model=fast -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffp-model=precise -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffp-model=strict -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcx-limited-range \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=BASIC %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fno-cx-limited-range \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=FULL %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcx-fortran-rules \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=IMPRVD %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fno-cx-fortran-rules \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=FULL %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -ffast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=BASIC %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcomplex-arithmetic=basic \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=BASIC %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcomplex-arithmetic=promoted \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=PRMTD %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcomplex-arithmetic=improved \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=IMPRVD %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -fcomplex-arithmetic=full \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=FULL %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -ffp-model=aggressive \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=BASIC %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -ffp-model=fast \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=PRMTD %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -ffp-model=precise \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=FULL %s

// RUN: %clang -### -Werror --target=x86_64 -fno-fast-math -ffp-model=strict \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=FULL %s

// WARN1: warning: overriding '-fcx-limited-range' option with '-fcx-fortran-rules' [-Woverriding-option]
// WARN2: warning: overriding '-fno-cx-limited-range' option with '-fcx-fortran-rules' [-Woverriding-option]
// WARN3: warning: overriding '-fcx-fortran-rules' option with '-fno-cx-limited-range' [-Woverriding-option]
// WARN4: warning: overriding '-fcx-limited-range' option with '-fno-cx-fortran-rules' [-Woverriding-option]
// WARN5: warning: overriding '-fcomplex-arithmetic=basic' option with '-fcomplex-arithmetic=improved' [-Woverriding-option]
// WARN6: warning: overriding '-fcx-limited-range' option with '-fcomplex-arithmetic=improved' [-Woverriding-option]
// WARN7: warning: overriding '-fcx-fortran-rules' option with '-fcomplex-arithmetic=basic' [-Woverriding-option]
// WARN14: overriding '-complex-range=promoted' option with '-fcx-limited-range' [-Woverriding-option]
// WARN17: warning: overriding '-fcomplex-arithmetic=full' option with '-fcomplex-arithmetic=basic' [-Woverriding-option]
// WARN20: warning: overriding '-fcx-fortran-rules' option with '-fcx-limited-range' [-Woverriding-option]
// WARN21: warning: overriding '-fcx-limited-range' option with '-fno-fast-math' [-Woverriding-option]
// WARN22: warning: overriding '-fcx-fortran-rules' option with '-fno-fast-math' [-Woverriding-option]
// WARN23: warning: overriding '-fcomplex-arithmetic=basic' option with '-fno-fast-math' [-Woverriding-option]
// WARN24: warning: overriding '-fcomplex-arithmetic=promoted' option with '-fno-fast-math' [-Woverriding-option]
// WARN25: warning: overriding '-fcomplex-arithmetic=improved' option with '-fno-fast-math' [-Woverriding-option]

// BASIC: -complex-range=basic
// FULL: -complex-range=full
// PRMTD: -complex-range=promoted
// BASIC-NOT: -complex-range=improved
// CHECK-NOT: -complex-range=basic
// IMPRVD: -complex-range=improved
// IMPRVD-NOT: -complex-range=basic
// CHECK-NOT: -complex-range=improved
// RANGE-NOT: -complex-range=

// ERR: error: unsupported argument 'foo' to option '-fcomplex-arithmetic='
