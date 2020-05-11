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

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx -mavx2 -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512dq -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2 -mavx512ifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512ifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVX %s
// AVX: "-target-feature" "+avx" "-target-feature" "+avx2" "-target-feature" "+avx512f" "-target-feature" "+avx512cd" "-target-feature" "+avx512er" "-target-feature" "+avx512pf" "-target-feature" "+avx512dq" "-target-feature" "+avx512bw" "-target-feature" "+avx512vl" "-target-feature" "+avx512vbmi" "-target-feature" "+avx512vbmi2" "-target-feature" "+avx512ifma"
// NO-AVX: "-target-feature" "-avx" "-target-feature" "-avx2" "-target-feature" "-avx512f" "-target-feature" "-avx512cd" "-target-feature" "-avx512er" "-target-feature" "-avx512pf" "-target-feature" "-avx512dq" "-target-feature" "-avx512bw" "-target-feature" "-avx512vl" "-target-feature" "-avx512vbmi" "-target-feature" "-avx512vbmi2" "-target-feature" "-avx512ifma"

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

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mwbnoinvd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=WBNOINVD %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-wbnoinvd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-WBNOINVD %s
// WBNOINVD: "-target-feature" "+wbnoinvd"
// NO-WBNOINVD: "-target-feature" "-wbnoinvd"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmovbe %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MOVBE %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-movbe %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MOVBE %s
// MOVBE: "-target-feature" "+movbe"
// NO-MOVBE: "-target-feature" "-movbe"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmpx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MPX %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-mpx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MPX %s
// MPX: the flag '-mmpx' has been deprecated and will be ignored
// NO-MPX: the flag '-mno-mpx' has been deprecated and will be ignored

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mshstk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CETSS %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-shstk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-CETSS %s
// CETSS: "-target-feature" "+shstk"
// NO-CETSS: "-target-feature" "-shstk"

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

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mvaes %s -### -o %t.o 2>&1 | FileCheck -check-prefix=VAES %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-vaes %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-VAES %s
// VAES: "-target-feature" "+vaes"
// NO-VAES: "-target-feature" "-vaes"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mgfni %s -### -o %t.o 2>&1 | FileCheck -check-prefix=GFNI %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-gfni %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-GFNI %s
// GFNI: "-target-feature" "+gfni"
// NO-GFNI: "-target-feature" "-gfni

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mvpclmulqdq %s -### -o %t.o 2>&1 | FileCheck -check-prefix=VPCLMULQDQ %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-vpclmulqdq %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-VPCLMULQDQ %s
// VPCLMULQDQ: "-target-feature" "+vpclmulqdq"
// NO-VPCLMULQDQ: "-target-feature" "-vpclmulqdq"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512bitalg %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BITALG %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx512bitalg %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-BITALG %s
// BITALG: "-target-feature" "+avx512bitalg"
// NO-BITALG: "-target-feature" "-avx512bitalg"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512vnni %s -### -o %t.o 2>&1 | FileCheck -check-prefix=VNNI %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx512vnni %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-VNNI %s
// VNNI: "-target-feature" "+avx512vnni"
// NO-VNNI: "-target-feature" "-avx512vnni"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512vbmi2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=VBMI2 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx512vbmi2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-VBMI2 %s
// VBMI2: "-target-feature" "+avx512vbmi2"
// NO-VBMI2: "-target-feature" "-avx512vbmi2"

// RUN: %clang -target i386-linux-gnu -mavx512vp2intersect %s -### -o %t.o 2>&1 | FileCheck -check-prefix=VP2INTERSECT %s
// RUN: %clang -target i386-linux-gnu -mno-avx512vp2intersect %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-VP2INTERSECT %s
// VP2INTERSECT: "-target-feature" "+avx512vp2intersect"
// NO-VP2INTERSECT: "-target-feature" "-avx512vp2intersect"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mrdpid %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RDPID %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-rdpid %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-RDPID %s
// RDPID: "-target-feature" "+rdpid"
// NO-RDPID: "-target-feature" "-rdpid"

// RUN: %clang -target i386-linux-gnu -mretpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RETPOLINE %s
// RUN: %clang -target i386-linux-gnu -mno-retpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-RETPOLINE %s
// RETPOLINE: "-target-feature" "+retpoline-indirect-calls" "-target-feature" "+retpoline-indirect-branches"
// NO-RETPOLINE-NOT: retpoline

// RUN: %clang -target i386-linux-gnu -mretpoline -mretpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RETPOLINE-EXTERNAL-THUNK %s
// RUN: %clang -target i386-linux-gnu -mretpoline -mno-retpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-RETPOLINE-EXTERNAL-THUNK %s
// RETPOLINE-EXTERNAL-THUNK: "-target-feature" "+retpoline-external-thunk"
// NO-RETPOLINE-EXTERNAL-THUNK: "-target-feature" "-retpoline-external-thunk"

// RUN: %clang -target i386-linux-gnu -mspeculative-load-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SLH %s
// RUN: %clang -target i386-linux-gnu -mretpoline -mspeculative-load-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=RETPOLINE %s
// RUN: %clang -target i386-linux-gnu -mno-speculative-load-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SLH %s
// SLH-NOT: retpoline
// SLH: "-target-feature" "+retpoline-indirect-calls"
// SLH-NOT: retpoline
// SLH: "-mspeculative-load-hardening"
// NO-SLH-NOT: retpoline

// RUN: %clang -target i386-linux-gnu -mlvi-cfi %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVICFI %s
// RUN: %clang -target i386-linux-gnu -mno-lvi-cfi %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-LVICFI %s
// LVICFI: "-target-feature" "+lvi-cfi"
// NO-LVICFI-NOT: lvi-cfi

// RUN: %clang -target i386-linux-gnu -mlvi-cfi -mspeculative-load-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVICFI-SLH %s
// LVICFI-SLH: error: invalid argument 'mspeculative-load-hardening' not allowed with 'mlvi-cfi'
// RUN: %clang -target i386-linux-gnu -mlvi-cfi -mretpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVICFI-RETPOLINE %s
// LVICFI-RETPOLINE: error: invalid argument 'mretpoline' not allowed with 'mlvi-cfi'
// RUN: %clang -target i386-linux-gnu -mlvi-cfi -mretpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVICFI-RETPOLINE-EXTERNAL-THUNK %s
// LVICFI-RETPOLINE-EXTERNAL-THUNK: error: invalid argument 'mretpoline-external-thunk' not allowed with 'mlvi-cfi'

// RUN: %clang -target i386-linux-gnu -mlvi-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVIHARDENING %s
// RUN: %clang -target i386-linux-gnu -mno-lvi-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-LVIHARDENING %s
// LVIHARDENING: "-target-feature" "+lvi-load-hardening" "-target-feature" "+lvi-cfi"
// NO-LVIHARDENING-NOT: lvi

// RUN: %clang -target i386-linux-gnu -mlvi-hardening -mspeculative-load-hardening %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVIHARDENING-SLH %s
// LVIHARDENING-SLH: error: invalid argument 'mspeculative-load-hardening' not allowed with 'mlvi-hardening'
// RUN: %clang -target i386-linux-gnu -mlvi-hardening -mretpoline %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVIHARDENING-RETPOLINE %s
// LVIHARDENING-RETPOLINE: error: invalid argument 'mretpoline' not allowed with 'mlvi-hardening'
// RUN: %clang -target i386-linux-gnu -mlvi-hardening -mretpoline-external-thunk %s -### -o %t.o 2>&1 | FileCheck -check-prefix=LVIHARDENING-RETPOLINE-EXTERNAL-THUNK %s
// LVIHARDENING-RETPOLINE-EXTERNAL-THUNK: error: invalid argument 'mretpoline-external-thunk' not allowed with 'mlvi-hardening'

// RUN: %clang -target i386-linux-gnu -mwaitpkg %s -### -o %t.o 2>&1 | FileCheck -check-prefix=WAITPKG %s
// RUN: %clang -target i386-linux-gnu -mno-waitpkg %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-WAITPKG %s
// WAITPKG: "-target-feature" "+waitpkg"
// NO-WAITPKG: "-target-feature" "-waitpkg"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmovdiri %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MOVDIRI %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-movdiri %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MOVDIRI %s
// MOVDIRI: "-target-feature" "+movdiri"
// NO-MOVDIRI: "-target-feature" "-movdiri"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mmovdir64b %s -### -o %t.o 2>&1 | FileCheck -check-prefix=MOVDIR64B %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-movdir64b %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-MOVDIR64B %s
// MOVDIR64B: "-target-feature" "+movdir64b"
// NO-MOVDIR64B: "-target-feature" "-movdir64b"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mpconfig %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PCONFIG %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-pconfig %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-PCONFIG %s
// PCONFIG: "-target-feature" "+pconfig"
// NO-PCONFIG: "-target-feature" "-pconfig"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mptwrite %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PTWRITE %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-ptwrite %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-PTWRITE %s
// PTWRITE: "-target-feature" "+ptwrite"
// NO-PTWRITE: "-target-feature" "-ptwrite"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -minvpcid %s -### -o %t.o 2>&1 | FileCheck -check-prefix=INVPCID %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-invpcid %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-INVPCID %s
// INVPCID: "-target-feature" "+invpcid"
// NO-INVPCID: "-target-feature" "-invpcid"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mavx512bf16 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX512BF16 %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-avx512bf16 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVX512BF16 %s
// AVX512BF16: "-target-feature" "+avx512bf16"
// NO-AVX512BF16: "-target-feature" "-avx512bf16"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -menqcmd %s -### -o %t.o 2>&1 | FileCheck --check-prefix=ENQCMD %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-enqcmd %s -### -o %t.o 2>&1 | FileCheck --check-prefix=NO-ENQCMD %s
// ENQCMD: "-target-feature" "+enqcmd"
// NO-ENQCMD: "-target-feature" "-enqcmd"

// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mvzeroupper %s -### -o %t.o 2>&1 | FileCheck --check-prefix=VZEROUPPER %s
// RUN: %clang -target i386-unknown-linux-gnu -march=i386 -mno-vzeroupper %s -### -o %t.o 2>&1 | FileCheck --check-prefix=NO-VZEROUPPER %s
// VZEROUPPER: "-target-feature" "+vzeroupper"
// NO-VZEROUPPER: "-target-feature" "-vzeroupper"
