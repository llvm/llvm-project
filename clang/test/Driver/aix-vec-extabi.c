// RUN:  %clang -### -target powerpc-unknown-aix -S -maltivec -mabi=vec-extabi %s 2>&1 | \
// RUN:  FileCheck %s

// CHECK: "-cc1"
// CHECK-SAME: "-mabi=vec-extabi"

// RUN:  %clang -### -target powerpc-unknown-aix -S -maltivec -mabi=vec-default %s 2>&1 | \
// RUN:  FileCheck %s --check-prefix=ERROR

// ERROR: The default Altivec ABI on AIX is not yet supported, use '-mabi=vec-extabi' for the extended Altivec ABI
