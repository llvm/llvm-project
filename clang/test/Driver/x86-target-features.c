// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mx87 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=X87 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-x87 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-X87 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -m80387 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=X87 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-80387 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-X87 %s
// X87: "-target-feature" "+x87"
// NO-X87: "-target-feature" "-x87"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmmx -m3dnow -m3dnowa %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MMX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-mmx -mno-3dnow -mno-3dnowa %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MMX %s
// MMX: "-target-feature" "+mmx" "-target-feature" "+3dnow" "-target-feature" "+3dnowa"
// NO-MMX: "-target-feature" "-mmx" "-target-feature" "-3dnow" "-target-feature" "-3dnowa"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -msse -msse2 -msse3 -mssse3 -msse4a -msse4.1 -msse4.2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SSE %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-sse -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4a -mno-sse4.1 -mno-sse4.2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SSE %s
// SSE: "-target-feature" "+sse" "-target-feature" "+sse2" "-target-feature" "+sse3" "-target-feature" "+ssse3" "-target-feature" "+sse4a" "-target-feature" "+sse4.1" "-target-feature" "+sse4.2"
// NO-SSE: "-target-feature" "-sse" "-target-feature" "-sse2" "-target-feature" "-sse3" "-target-feature" "-ssse3" "-target-feature" "-sse4a" "-target-feature" "-sse4.1" "-target-feature" "-sse4.2"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -msse4 -maes %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SSE4-AES %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-sse4 -mno-aes %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SSE4-AES %s
// SSE4-AES: "-target-feature" "+sse4.2" "-target-feature" "+aes"
// NO-SSE4-AES: "-target-feature" "-sse4.1" "-target-feature" "-aes"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx -mavx2 -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512dq -mavx512bw -mavx512vl -mavx512vbmi -mavx512ifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512vbmi -mno-avx512ifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVX %s
// AVX: "-target-feature" "+avx" "-target-feature" "+avx2" "-target-feature" "+avx512f" "-target-feature" "+avx512cd" "-target-feature" "+avx512er" "-target-feature" "+avx512pf" "-target-feature" "+avx512dq" "-target-feature" "+avx512bw" "-target-feature" "+avx512vl" "-target-feature" "+avx512vbmi" "-target-feature" "+avx512ifma"
// NO-AVX: "-target-feature" "-avx" "-target-feature" "-avx2" "-target-feature" "-avx512f" "-target-feature" "-avx512cd" "-target-feature" "-avx512er" "-target-feature" "-avx512pf" "-target-feature" "-avx512dq" "-target-feature" "-avx512bw" "-target-feature" "-avx512vl" "-target-feature" "-avx512vbmi" "-target-feature" "-avx512ifma"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mpclmul -mrdrnd -mfsgsbase -mbmi -mbmi2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BMI %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-pclmul -mno-rdrnd -mno-fsgsbase -mno-bmi -mno-bmi2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-BMI %s
// BMI: "-target-feature" "+pclmul" "-target-feature" "+rdrnd" "-target-feature" "+fsgsbase" "-target-feature" "+bmi" "-target-feature" "+bmi2"
// NO-BMI: "-target-feature" "-pclmul" "-target-feature" "-rdrnd" "-target-feature" "-fsgsbase" "-target-feature" "-bmi" "-target-feature" "-bmi2"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mlzcnt -mpopcnt -mtbm -mfma -mfma4 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=FMA %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-lzcnt -mno-popcnt -mno-tbm -mno-fma -mno-fma4 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-FMA %s
// FMA: "-target-feature" "+lzcnt" "-target-feature" "+popcnt" "-target-feature" "+tbm" "-target-feature" "+fma" "-target-feature" "+fma4"
// NO-FMA: "-target-feature" "-lzcnt" "-target-feature" "-popcnt" "-target-feature" "-tbm" "-target-feature" "-fma" "-target-feature" "-fma4"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mxop -mf16c -mrtm -mprfchw -mrdseed %s -### -o %t.o 2>&1 | FileCheck -check-prefix=XOP %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-xop -mno-f16c -mno-rtm -mno-prfchw -mno-rdseed %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-XOP %s
// XOP: "-target-feature" "+xop" "-target-feature" "+f16c" "-target-feature" "+rtm" "-target-feature" "+prfchw" "-target-feature" "+rdseed"
// NO-XOP: "-target-feature" "-xop" "-target-feature" "-f16c" "-target-feature" "-rtm" "-target-feature" "-prfchw" "-target-feature" "-rdseed"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -msha -mpku -madx -mcx16 -mfxsr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SHA %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-sha -mno-pku -mno-adx -mno-cx16 -mno-fxsr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SHA %s
// SHA: "-target-feature" "+sha" "-target-feature" "+pku" "-target-feature" "+adx" "-target-feature" "+cx16" "-target-feature" "+fxsr"
// NO-SHA: "-target-feature" "-sha" "-target-feature" "-pku" "-target-feature" "-adx" "-target-feature" "-cx16" "-target-feature" "-fxsr"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mxsave -mxsaveopt -mxsavec -mxsaves %s -### -o %t.o 2>&1 | FileCheck -check-prefix=XSAVE %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-xsave -mno-xsaveopt -mno-xsavec -mno-xsaves %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-XSAVE %s
// XSAVE: "-target-feature" "+xsave" "-target-feature" "+xsaveopt" "-target-feature" "+xsavec" "-target-feature" "+xsaves"
// NO-XSAVE: "-target-feature" "-xsave" "-target-feature" "-xsaveopt" "-target-feature" "-xsavec" "-target-feature" "-xsaves"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mclflushopt %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CLFLUSHOPT %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-clflushopt %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-CLFLUSHOPT %s
// CLFLUSHOPT: "-target-feature" "+clflushopt"
// NO-CLFLUSHOPT: "-target-feature" "-clflushopt"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mclwb %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CLWB %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-clwb %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-CLWB %s
// CLWB: "-target-feature" "+clwb"
// NO-CLWB: "-target-feature" "-clwb"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmovbe %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MOVBE %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-movbe %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MOVBE %s
// MOVBE: "-target-feature" "+movbe"
// NO-MOVBE: "-target-feature" "-movbe"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmpx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MPX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-mpx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MPX %s
// MPX: "-target-feature" "+mpx"
// NO-MPX: "-target-feature" "-mpx"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -msgx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SGX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-sgx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SGX %s
// SGX: "-target-feature" "+sgx"
// NO-SGX: "-target-feature" "-sgx"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mprefetchwt1 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PREFETCHWT1 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-prefetchwt1 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-PREFETCHWT1 %s
// PREFETCHWT1: "-target-feature" "+prefetchwt1"
// NO-PREFETCHWT1: "-target-feature" "-prefetchwt1"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mclzero %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CLZERO %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-clzero %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-CLZERO %s
// CLZERO: "-target-feature" "+clzero"
// NO-CLZERO: "-target-feature" "-clzero"

// RUN: %clang -target i386-linux-gnu -mretpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RETPOLINE %s
// RUN: %clang -target i386-linux-gnu -mno-retpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-RETPOLINE %s
// RETPOLINE: "-target-feature" "+retpoline"
// NO-RETPOLINE: "-target-feature" "-retpoline"

// RUN: %clang -target i386-linux-gnu -mretpoline -mretpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RETPOLINE-EXTERNAL-THUNK %s
// RUN: %clang -target i386-linux-gnu -mretpoline -mno-retpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-RETPOLINE-EXTERNAL-THUNK %s
// RETPOLINE-EXTERNAL-THUNK: "-target-feature" "+retpoline-external-thunk"
// NO-RETPOLINE-EXTERNAL-THUNK: "-target-feature" "-retpoline-external-thunk"
