// Verify error message is emitted for `-gsplit-dwarf` on AIX 
// as it's unsupported at the moment.

// RUN: not %clang -target powerpc-ibm-aix -gdwarf-4 -gsplit-dwarf %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=UNSUP_OPT_AIX
// RUN: not %clang -target powerpc64-ibm-aix -gdwarf-4 -gsplit-dwarf %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=UNSUP_OPT_AIX64

// UNSUP_OPT_AIX: error: unsupported option '-gsplit-dwarf' for target 'powerpc-ibm-aix'
// UNSUP_OPT_AIX64: error: unsupported option '-gsplit-dwarf' for target 'powerpc64-ibm-aix'

int main(){return 0;}
