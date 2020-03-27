// Checks that cuda compilation does the right thing when passed
// -fcuda-flush-denormals-to-zero. This should be translated to
// -fdenormal-fp-math-f32=preserve-sign

// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c -march=haswell --cuda-gpu-arch=sm_20 -fcuda-flush-denormals-to-zero -nocudainc -nocudalib %s 2>&1 | FileCheck -check-prefix=FTZ %s
// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c -march=haswell --cuda-gpu-arch=sm_20 -fno-cuda-flush-denormals-to-zero -nocudainc -nocudalib %s 2>&1 | FileCheck -check-prefix=NOFTZ %s
// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c -march=haswell --cuda-gpu-arch=sm_70 -fcuda-flush-denormals-to-zero -nocudainc -nocudalib %s 2>&1 | FileCheck -check-prefix=FTZ %s
// RUN: %clang -no-canonical-prefixes -### -target x86_64-linux-gnu -c -march=haswell --cuda-gpu-arch=sm_70 -fno-cuda-flush-denormals-to-zero -nocudainc -nocudalib %s 2>&1 | FileCheck -check-prefix=NOFTZ %s

// CPUFTZ-NOT: -fdenormal-fp-math

// FTZ-NOT: -fdenormal-fp-math-f32=
// FTZ: "-fdenormal-fp-math-f32=preserve-sign,preserve-sign"

// The default of ieee is omitted
// NOFTZ-NOT: "-fdenormal-fp-math"
// NOFTZ-NOT: "-fdenormal-fp-math-f32"
