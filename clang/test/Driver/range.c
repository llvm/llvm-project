// Test range options for complex multiplication and division.

// RUN: %clang -### -target x86_64 -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -fcx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=BASIC %s

// RUN: %clang -### -target x86_64 -fno-cx-limited-range -fno-cx-fortran-rules \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcx-fortran-rules -c %s 2>&1 \
// RUN:   -fno-cx-fortran-rules | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck  %s

// RUN: %clang -### -target x86_64 -fcx-limited-range -fno-cx-limited-range \
// RUN: -c %s 2>&1 | FileCheck --check-prefix=FULL %s

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
// RUN: -fcomplex-arithmetic=improved  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=IMPRVD %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=promoted \
// RUN: -fcomplex-arithmetic=full  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=FULL %s

// RUN: %clang -### -target x86_64 -fcomplex-arithmetic=full \
// RUN: -fcomplex-arithmetic=basic  -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=BASIC %s

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
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -fno-cx-limited-range -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcx-fortran-rules -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -fno-cx-fortran-rules -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### -Werror --target=x86_64 -ffast-math -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=basic -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=promoted -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

// RUN: %clang -### --target=x86_64 -fcomplex-arithmetic=improved -fno-fast-math \
// RUN:   -c %s 2>&1 | FileCheck --check-prefixes=RANGE %s

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
