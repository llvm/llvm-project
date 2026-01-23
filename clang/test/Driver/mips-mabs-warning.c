// REQUIRES: mips-registered-target
// RUN: %clang -c --target=mips-unknown-gnu -mcpu=mips32 -mabs=2008 %s 2>&1 | FileCheck -check-prefix=NO2008 %s
// NO2008: warning: ignoring '-mabs=2008' option because the 'mips32' architecture does not support it [-Wunsupported-abs]

// RUN: %clang -c --target=mips-unknown-gnu -mcpu=mips32r6 -mabs=legacy %s 2>&1 | FileCheck -check-prefix=NOLEGACY %s
// NOLEGACY: warning: ignoring '-mabs=legacy' option because the 'mips32r6' architecture does not support it [-Wunsupported-abs]
