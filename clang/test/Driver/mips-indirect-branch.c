// -mindirect-jump=hazard -mips32
// RUN: not %clang --target=mips-unknown-linux-gnu -mips32 -### -c %s \
// RUN:        -mindirect-jump=hazard 2>&1 | FileCheck %s --check-prefix=MIPS32
// MIPS32: error: '-mindirect-jump=hazard' is unsupported with the 'mips32' architecture

// -mindirect-jump=hazard -mmicromips
// RUN: not %clang --target=mips-unknown-linux-gnu -mmicromips -### -c %s \
// RUN:        -mindirect-jump=hazard 2>&1 | FileCheck %s --check-prefix=MICROMIPS
// MICROMIPS: error: '-mindirect-jump=hazard' is unsupported with the 'micromips' architecture

// -mindirect-jump=hazard -mips16
// RUN: not %clang --target=mips-unknown-linux-gnu -mips16 -### -c %s \
// RUN:        -mindirect-jump=hazard 2>&1 | FileCheck %s --check-prefix=MIPS16
// MIPS16: error: '-mindirect-jump=hazard' is unsupported with the 'mips16' architecture

// RUN: not %clang --target=mips-unknown-linux-gnu  -### -c %s \
// RUN:        -mindirect-jump=retopline 2>&1 | FileCheck %s --check-prefix=RETOPLINE
// RETOPLINE: error: unknown '-mindirect-jump=' option 'retopline'

// RUN: not %clang --target=mips-unknown-linux-gnu  -### -mips32 -c %s \
// RUN:        -mindirect-jump=retopline 2>&1 | FileCheck %s --check-prefix=MIXED
// MIXED: error: unknown '-mindirect-jump=' option 'retopline'
