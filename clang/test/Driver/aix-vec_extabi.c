// RUN:  %clang -### -target powerpc-unknown-aix -S %s 2>&1 | \
// RUN:  FileCheck %s --implicit-check-not=vec-extabi
// RUN:  %clang -### -target powerpc-unknown-aix -S -maltivec %s 2>&1 | \
// RUN:  FileCheck %s --implicit-check-not=vec-extabi
// RUN:  %clang -### -target powerpc-unknown-aix -S -maltivec -mabi=vec-default %s 2>&1 | \
// RUN:  FileCheck %s --check-prefix=DFLTABI --implicit-check-not=vec-extabi
// RUN:  %clang -### -target powerpc-unknown-aix -S -mabi=vec-extabi %s 2>&1 | \
// RUN:  FileCheck %s --check-prefix=EXTABI
// RUN:  %clang -### -target powerpc-unknown-aix -S -maltivec -mabi=vec-extabi %s 2>&1 | \
// RUN:  FileCheck %s --check-prefix=EXTABI
/
// EXTABI:       "-cc1"
// EXTABI-SAME:  "-mabi=vec-extabi"

// DFLTABI:      "-cc1"
// DFLTABI-NOT:  "-mabi=vec-extabi"
