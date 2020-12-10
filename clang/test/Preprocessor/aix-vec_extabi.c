// RUN: %clang -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -E -dM -maltivec -mabi=vec-extabi %s -o - 2>&1 \
// RUN:   | FileCheck %s -check-prefix=EXTABI
// RUN: %clang  -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -E -dM -maltivec -mabi=vec-extabi %s -o - 2>&1 \
// RUN:   | FileCheck %s -check-prefix=EXTABI
// RUN: not %clang  -target powerpc-ibm-aix-xcoff -mcpu=pwr8 -E -dM -maltivec -mabi=vec-default %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DFLTABI
// RUN: not %clang -target powerpc64-ibm-aix-xcoff -mcpu=pwr8 -E -dM -maltivec -mabi=vec-default %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DFLTABI


// EXTABI:  #define __EXTABI__
// DFLTABI: The default Altivec ABI on AIX is not yet supported, use '-mabi=vec-extabi' for the extended Altivec ABI
