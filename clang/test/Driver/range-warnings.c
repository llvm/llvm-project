// Test overriding warnings about complex range.
// range.c tests the settings of -complex-range=, and this test covers
// all warnings related to complex range.

// Clang options related to complex range are as follows:
//   -f[no-]fast-math
//   -f[no-]cx-limited-range
//   -f[no-]cx-fortran-rules
//   -fcomplex-arithmetic=[full|improved|promoted|basic]
//   -ffp-model=[strict|precise|fast|aggressive]

// Emit warnings about overriding when options implying different
// complex ranges are specified. However, warnings are not emitted in
// the following cases:
// (a) When the positive/negative form or a different value of the same
//     option is specified. 
//       Example: 
//          `-ffast-math -fno-fast-math`
//          `-fcx-limited-range -fno-cx-limited-range`
//          `-fcx-fortran-rules -fno-cx-fortran-rules`
//          `-fcomplex-arithmetic=full -fcomplex-arithmetic=improved`
//          `-ffp-model=strict -ffp-model=aggressive`
//
// (b) When -ffp-model= is overridden by -f[no-]fast-math. 
//       Example:
//          `-ffp-model=fast -fno-fast-math`
//          `-ffp-model=strict -ffast-math`


// RUN: %clang -### -Werror -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffast-math -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-FULL-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffast-math -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -ffast-math -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,FAST-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffast-math -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-fast-math -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcx-limited-range -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFAST-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcx-limited-range -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcx-limited-range -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-FULL-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcx-limited-range -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcx-limited-range -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-limited-range -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,LIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcx-limited-range -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-limited-range -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-limited-range -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-limited-range -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-limited-range -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-limited-range -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-limited-range -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-limited-range -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-limited-range -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,NOLIM-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFAST-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcx-fortran-rules -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcx-fortran-rules -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-FULL-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcx-fortran-rules -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcx-fortran-rules -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fcx-fortran-rules -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,FORT-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-fortran-rules -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-fortran-rules -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-fortran-rules -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-fortran-rules -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-fortran-rules -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fno-cx-fortran-rules -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fno-cx-fortran-rules -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -fno-cx-fortran-rules -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,NOFORT-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=full -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=full -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=full -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=full -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=full -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=full -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,ARITH-FULL-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFAST-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=improved -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=improved -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=improved -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=improved -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=improved -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,ARITH-IMPROVED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FAST-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFAST-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=promoted -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=promoted -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=promoted -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=promoted -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=promoted -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-AGGRESSIVE-OVERRIDING,ARITH-PROMOTED-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFAST-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-STRICT-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-PRECISE-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -fcomplex-arithmetic=basic -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=MODEL-FAST-OVERRIDING,ARITH-BASIC-OVERRIDDEN %s

// RUN: %clang -### -Werror -fcomplex-arithmetic=basic -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=strict -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,MODEL-STRICT-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=strict -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,MODEL-STRICT-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=strict -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=strict -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,MODEL-STRICT-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=strict -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,MODEL-STRICT-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=strict -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,MODEL-STRICT-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=strict -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=strict -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=precise -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=precise -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=precise -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,MODEL-PRECISE-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=precise -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=precise -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,MODEL-PRECISE-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=precise -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=precise -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=precise -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,MODEL-PRECISE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=precise -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,MODEL-PRECISE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=precise -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,MODEL-PRECISE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=precise -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=precise -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=precise -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=fast -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=fast -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=fast -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=LIM-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-FULL-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=fast -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=fast -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-BASIC-OVERRIDING,MODEL-FAST-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=fast -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=fast -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=fast -ffp-model=aggressive -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=aggressive -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=aggressive -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -Werror -ffp-model=aggressive -fcx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=aggressive -fno-cx-limited-range -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOLIM-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=aggressive -fcx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=FORT-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=aggressive -fno-cx-fortran-rules -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NOFORT-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=aggressive -fcomplex-arithmetic=full -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-FULL-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=aggressive -fcomplex-arithmetic=improved -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-IMPROVED-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -ffp-model=aggressive -fcomplex-arithmetic=promoted -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=ARITH-PROMOTED-OVERRIDING,MODEL-AGGRESSIVE-OVERRIDDEN %s

// RUN: %clang -### -Werror -ffp-model=aggressive -fcomplex-arithmetic=basic -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=aggressive -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=aggressive -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s

// RUN: %clang -### -ffp-model=aggressive -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=NO-OVR-WARN %s


// NO-OVR-WARN-NOT: [-Woverriding-complex-range]

// FAST-OVERRIDING: warning: '-ffast-math' sets complex range to "basic"
// NOFAST-OVERRIDING: warning: '-fno-fast-math' sets complex range to "none"
// LIM-OVERRIDING: warning: '-fcx-limited-range' sets complex range to "basic"
// NOLIM-OVERRIDING: warning: '-fno-cx-limited-range' sets complex range to "full"
// FORT-OVERRIDING: warning: '-fcx-fortran-rules' sets complex range to "improved"
// NOFORT-OVERRIDING: warning: '-fno-cx-fortran-rules' sets complex range to "full"
// ARITH-FULL-OVERRIDING: warning: '-fcomplex-arithmetic=full' sets complex range to "full"
// ARITH-IMPROVED-OVERRIDING: warning: '-fcomplex-arithmetic=improved' sets complex range to "improved"
// ARITH-PROMOTED-OVERRIDING: warning: '-fcomplex-arithmetic=promoted' sets complex range to "promoted"
// ARITH-BASIC-OVERRIDING: warning: '-fcomplex-arithmetic=basic' sets complex range to "basic"
// MODEL-STRICT-OVERRIDING: warning: '-ffp-model=strict' sets complex range to "full"
// MODEL-PRECISE-OVERRIDING: warning: '-ffp-model=precise' sets complex range to "full"
// MODEL-FAST-OVERRIDING: warning: '-ffp-model=fast' sets complex range to "promoted"
// MODEL-AGGRESSIVE-OVERRIDING: warning: '-ffp-model=aggressive' sets complex range to "basic"

// FAST-OVERRIDDEN: overriding the setting of "basic" that was implied by '-ffast-math' [-Woverriding-complex-range]
// LIM-OVERRIDDEN: overriding the setting of "basic" that was implied by '-fcx-limited-range' [-Woverriding-complex-range]
// NOLIM-OVERRIDDEN: overriding the setting of "full" that was implied by '-fno-cx-limited-range' [-Woverriding-complex-range]
// FORT-OVERRIDDEN: overriding the setting of "improved" that was implied by '-fcx-fortran-rules' [-Woverriding-complex-range]
// NOFORT-OVERRIDDEN: overriding the setting of "full" that was implied by '-fno-cx-fortran-rules' [-Woverriding-complex-range]
// ARITH-FULL-OVERRIDDEN: overriding the setting of "full" that was implied by '-fcomplex-arithmetic=full' [-Woverriding-complex-range]
// ARITH-IMPROVED-OVERRIDDEN: overriding the setting of "improved" that was implied by '-fcomplex-arithmetic=improved' [-Woverriding-complex-range]
// ARITH-PROMOTED-OVERRIDDEN: overriding the setting of "promoted" that was implied by '-fcomplex-arithmetic=promoted' [-Woverriding-complex-range]
// ARITH-BASIC-OVERRIDDEN: overriding the setting of "basic" that was implied by '-fcomplex-arithmetic=basic' [-Woverriding-complex-range]
// MODEL-STRICT-OVERRIDDEN: overriding the setting of "full" that was implied by '-ffp-model=strict' [-Woverriding-complex-range]
// MODEL-PRECISE-OVERRIDDEN: overriding the setting of "full" that was implied by '-ffp-model=precise' [-Woverriding-complex-range]
// MODEL-FAST-OVERRIDDEN: overriding the setting of "promoted" that was implied by '-ffp-model=fast' [-Woverriding-complex-range]
// MODEL-AGGRESSIVE-OVERRIDDEN: overriding the setting of "basic" that was implied by '-ffp-model=aggressive' [-Woverriding-complex-range]
