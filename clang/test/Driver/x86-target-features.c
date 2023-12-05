// RUN: %clang --target=i386 -march=i386 -mx87 %s -### 2>&1 | FileCheck -check-prefix=X87 %s
// RUN: %clang --target=i386 -march=i386 -mno-x87 %s -### 2>&1 | FileCheck -check-prefix=NO-X87 %s
// RUN: %clang --target=i386 -march=i386 -m80387 %s -### 2>&1 | FileCheck -check-prefix=X87 %s
// RUN: %clang --target=i386 -march=i386 -mno-80387 %s -### 2>&1 | FileCheck -check-prefix=NO-X87 %s
// RUN: %clang --target=i386 -march=i386 -mno-fp-ret-in-387 %s -### 2>&1 | FileCheck -check-prefix=NO-X87 %s
// X87: "-target-feature" "+x87"
// NO-X87: "-target-feature" "-x87"

// RUN: %clang --target=i386 -march=i386 -mmmx -m3dnow -m3dnowa %s -### 2>&1 | FileCheck -check-prefix=MMX %s
// RUN: %clang --target=i386 -march=i386 -mno-mmx -mno-3dnow -mno-3dnowa %s -### 2>&1 | FileCheck -check-prefix=NO-MMX %s
// MMX: "-target-feature" "+mmx" "-target-feature" "+3dnow" "-target-feature" "+3dnowa"
// NO-MMX: "-target-feature" "-mmx" "-target-feature" "-3dnow" "-target-feature" "-3dnowa"

// RUN: %clang --target=i386 -march=i386 -msse -msse2 -msse3 -mssse3 -msse4a -msse4.1 -msse4.2 %s -### 2>&1 | FileCheck -check-prefix=SSE %s
// RUN: %clang --target=i386 -march=i386 -mno-sse -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4a -mno-sse4.1 -mno-sse4.2 %s -### 2>&1 | FileCheck -check-prefix=NO-SSE %s
// SSE: "-target-feature" "+sse" "-target-feature" "+sse2" "-target-feature" "+sse3" "-target-feature" "+ssse3" "-target-feature" "+sse4a" "-target-feature" "+sse4.1" "-target-feature" "+sse4.2"
// NO-SSE: "-target-feature" "-sse" "-target-feature" "-sse2" "-target-feature" "-sse3" "-target-feature" "-ssse3" "-target-feature" "-sse4a" "-target-feature" "-sse4.1" "-target-feature" "-sse4.2"

// RUN: %clang --target=i386 -march=i386 -msse4 -maes %s -### 2>&1 | FileCheck -check-prefix=SSE4-AES %s
// RUN: %clang --target=i386 -march=i386 -mno-sse4 -mno-aes %s -### 2>&1 | FileCheck -check-prefix=NO-SSE4-AES %s
// SSE4-AES: "-target-feature" "+sse4.2" "-target-feature" "+aes"
// NO-SSE4-AES: "-target-feature" "-sse4.1" "-target-feature" "-aes"

// RUN: %clang --target=i386 -march=i386 -mavx -mavx2 -mavx512f -mavx512cd -mavx512er -mavx512pf -mavx512dq -mavx512bw -mavx512vl -mavx512vbmi -mavx512vbmi2 -mavx512ifma %s -### 2>&1 | FileCheck -check-prefix=AVX %s
// RUN: %clang --target=i386 -march=i386 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512cd -mno-avx512er -mno-avx512pf -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512ifma %s -### 2>&1 | FileCheck -check-prefix=NO-AVX %s
// AVX: "-target-feature" "+avx" "-target-feature" "+avx2" "-target-feature" "+avx512f" "-target-feature" "+avx512cd" "-target-feature" "+avx512er" "-target-feature" "+avx512pf" "-target-feature" "+avx512dq" "-target-feature" "+avx512bw" "-target-feature" "+avx512vl" "-target-feature" "+avx512vbmi" "-target-feature" "+avx512vbmi2" "-target-feature" "+avx512ifma"
// NO-AVX: "-target-feature" "-avx" "-target-feature" "-avx2" "-target-feature" "-avx512f" "-target-feature" "-avx512cd" "-target-feature" "-avx512er" "-target-feature" "-avx512pf" "-target-feature" "-avx512dq" "-target-feature" "-avx512bw" "-target-feature" "-avx512vl" "-target-feature" "-avx512vbmi" "-target-feature" "-avx512vbmi2" "-target-feature" "-avx512ifma"

// RUN: %clang --target=i386 -march=i386 -mpclmul -mrdrnd -mfsgsbase -mbmi -mbmi2 %s -### 2>&1 | FileCheck -check-prefix=BMI %s
// RUN: %clang --target=i386 -march=i386 -mno-pclmul -mno-rdrnd -mno-fsgsbase -mno-bmi -mno-bmi2 %s -### 2>&1 | FileCheck -check-prefix=NO-BMI %s
// BMI: "-target-feature" "+pclmul" "-target-feature" "+rdrnd" "-target-feature" "+fsgsbase" "-target-feature" "+bmi" "-target-feature" "+bmi2"
// NO-BMI: "-target-feature" "-pclmul" "-target-feature" "-rdrnd" "-target-feature" "-fsgsbase" "-target-feature" "-bmi" "-target-feature" "-bmi2"

// RUN: %clang --target=i386 -march=i386 -mlzcnt -mpopcnt -mtbm -mfma -mfma4 %s -### 2>&1 | FileCheck -check-prefix=FMA %s
// RUN: %clang --target=i386 -march=i386 -mno-lzcnt -mno-popcnt -mno-tbm -mno-fma -mno-fma4 %s -### 2>&1 | FileCheck -check-prefix=NO-FMA %s
// FMA: "-target-feature" "+lzcnt" "-target-feature" "+popcnt" "-target-feature" "+tbm" "-target-feature" "+fma" "-target-feature" "+fma4"
// NO-FMA: "-target-feature" "-lzcnt" "-target-feature" "-popcnt" "-target-feature" "-tbm" "-target-feature" "-fma" "-target-feature" "-fma4"

// RUN: %clang --target=i386 -march=i386 -mxop -mf16c -mrtm -mprfchw -mrdseed %s -### 2>&1 | FileCheck -check-prefix=XOP %s
// RUN: %clang --target=i386 -march=i386 -mno-xop -mno-f16c -mno-rtm -mno-prfchw -mno-rdseed %s -### 2>&1 | FileCheck -check-prefix=NO-XOP %s
// XOP: "-target-feature" "+xop" "-target-feature" "+f16c" "-target-feature" "+rtm" "-target-feature" "+prfchw" "-target-feature" "+rdseed"
// NO-XOP: "-target-feature" "-xop" "-target-feature" "-f16c" "-target-feature" "-rtm" "-target-feature" "-prfchw" "-target-feature" "-rdseed"

// RUN: %clang --target=i386 -march=i386 -msha -mpku -madx -mcx16 -mfxsr %s -### 2>&1 | FileCheck -check-prefix=SHA %s
// RUN: %clang --target=i386 -march=i386 -mno-sha -mno-pku -mno-adx -mno-cx16 -mno-fxsr %s -### 2>&1 | FileCheck -check-prefix=NO-SHA %s
// SHA: "-target-feature" "+sha" "-target-feature" "+pku" "-target-feature" "+adx" "-target-feature" "+cx16" "-target-feature" "+fxsr"
// NO-SHA: "-target-feature" "-sha" "-target-feature" "-pku" "-target-feature" "-adx" "-target-feature" "-cx16" "-target-feature" "-fxsr"

// RUN: %clang --target=i386 -march=i386 -mxsave -mxsaveopt -mxsavec -mxsaves %s -### 2>&1 | FileCheck -check-prefix=XSAVE %s
// RUN: %clang --target=i386 -march=i386 -mno-xsave -mno-xsaveopt -mno-xsavec -mno-xsaves %s -### 2>&1 | FileCheck -check-prefix=NO-XSAVE %s
// XSAVE: "-target-feature" "+xsave" "-target-feature" "+xsaveopt" "-target-feature" "+xsavec" "-target-feature" "+xsaves"
// NO-XSAVE: "-target-feature" "-xsave" "-target-feature" "-xsaveopt" "-target-feature" "-xsavec" "-target-feature" "-xsaves"

// RUN: %clang --target=i386 -march=i386 -mclflushopt %s -### 2>&1 | FileCheck -check-prefix=CLFLUSHOPT %s
// RUN: %clang --target=i386 -march=i386 -mno-clflushopt %s -### 2>&1 | FileCheck -check-prefix=NO-CLFLUSHOPT %s
// CLFLUSHOPT: "-target-feature" "+clflushopt"
// NO-CLFLUSHOPT: "-target-feature" "-clflushopt"

// RUN: %clang --target=i386 -march=i386 -mclwb %s -### 2>&1 | FileCheck -check-prefix=CLWB %s
// RUN: %clang --target=i386 -march=i386 -mno-clwb %s -### 2>&1 | FileCheck -check-prefix=NO-CLWB %s
// CLWB: "-target-feature" "+clwb"
// NO-CLWB: "-target-feature" "-clwb"

// RUN: %clang --target=i386 -march=i386 -mwbnoinvd %s -### 2>&1 | FileCheck -check-prefix=WBNOINVD %s
// RUN: %clang --target=i386 -march=i386 -mno-wbnoinvd %s -### 2>&1 | FileCheck -check-prefix=NO-WBNOINVD %s
// WBNOINVD: "-target-feature" "+wbnoinvd"
// NO-WBNOINVD: "-target-feature" "-wbnoinvd"

// RUN: %clang --target=i386 -march=i386 -mmovbe %s -### 2>&1 | FileCheck -check-prefix=MOVBE %s
// RUN: %clang --target=i386 -march=i386 -mno-movbe %s -### 2>&1 | FileCheck -check-prefix=NO-MOVBE %s
// MOVBE: "-target-feature" "+movbe"
// NO-MOVBE: "-target-feature" "-movbe"

// RUN: %clang --target=i386 -march=i386 -mmpx %s -### 2>&1 | FileCheck -check-prefix=MPX %s
// RUN: %clang --target=i386 -march=i386 -mno-mpx %s -### 2>&1 | FileCheck -check-prefix=NO-MPX %s
// MPX: the flag '-mmpx' has been deprecated and will be ignored
// NO-MPX: the flag '-mno-mpx' has been deprecated and will be ignored

// RUN: %clang --target=i386 -march=i386 -mshstk %s -### 2>&1 | FileCheck -check-prefix=CETSS %s
// RUN: %clang --target=i386 -march=i386 -mno-shstk %s -### 2>&1 | FileCheck -check-prefix=NO-CETSS %s
// CETSS: "-target-feature" "+shstk"
// NO-CETSS: "-target-feature" "-shstk"

// RUN: %clang --target=i386 -march=i386 -msgx %s -### 2>&1 | FileCheck -check-prefix=SGX %s
// RUN: %clang --target=i386 -march=i386 -mno-sgx %s -### 2>&1 | FileCheck -check-prefix=NO-SGX %s
// SGX: "-target-feature" "+sgx"
// NO-SGX: "-target-feature" "-sgx"

// RUN: %clang --target=i386 -march=i386 -mprefetchwt1 %s -### 2>&1 | FileCheck -check-prefix=PREFETCHWT1 %s
// RUN: %clang --target=i386 -march=i386 -mno-prefetchwt1 %s -### 2>&1 | FileCheck -check-prefix=NO-PREFETCHWT1 %s
// PREFETCHWT1: "-target-feature" "+prefetchwt1"
// NO-PREFETCHWT1: "-target-feature" "-prefetchwt1"

// RUN: %clang --target=i386 -march=i386 -mprefetchi %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PREFETCHI %s
// RUN: %clang --target=i386 -march=i386 -mno-prefetchi %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-PREFETCHI %s
// PREFETCHI: "-target-feature" "+prefetchi"
// NO-PREFETCHI: "-target-feature" "-prefetchi"

// RUN: %clang --target=i386 -march=i386 -mclzero %s -### 2>&1 | FileCheck -check-prefix=CLZERO %s
// RUN: %clang --target=i386 -march=i386 -mno-clzero %s -### 2>&1 | FileCheck -check-prefix=NO-CLZERO %s
// CLZERO: "-target-feature" "+clzero"
// NO-CLZERO: "-target-feature" "-clzero"

// RUN: %clang --target=i386 -march=i386 -mvaes %s -### 2>&1 | FileCheck -check-prefix=VAES %s
// RUN: %clang --target=i386 -march=i386 -mno-vaes %s -### 2>&1 | FileCheck -check-prefix=NO-VAES %s
// VAES: "-target-feature" "+vaes"
// NO-VAES: "-target-feature" "-vaes"

// RUN: %clang --target=i386 -march=i386 -mgfni %s -### 2>&1 | FileCheck -check-prefix=GFNI %s
// RUN: %clang --target=i386 -march=i386 -mno-gfni %s -### 2>&1 | FileCheck -check-prefix=NO-GFNI %s
// GFNI: "-target-feature" "+gfni"
// NO-GFNI: "-target-feature" "-gfni

// RUN: %clang --target=i386 -march=i386 -mvpclmulqdq %s -### 2>&1 | FileCheck -check-prefix=VPCLMULQDQ %s
// RUN: %clang --target=i386 -march=i386 -mno-vpclmulqdq %s -### 2>&1 | FileCheck -check-prefix=NO-VPCLMULQDQ %s
// VPCLMULQDQ: "-target-feature" "+vpclmulqdq"
// NO-VPCLMULQDQ: "-target-feature" "-vpclmulqdq"

// RUN: %clang --target=i386 -march=i386 -mavx512bitalg %s -### 2>&1 | FileCheck -check-prefix=BITALG %s
// RUN: %clang --target=i386 -march=i386 -mno-avx512bitalg %s -### 2>&1 | FileCheck -check-prefix=NO-BITALG %s
// BITALG: "-target-feature" "+avx512bitalg"
// NO-BITALG: "-target-feature" "-avx512bitalg"

// RUN: %clang --target=i386 -march=i386 -mavx512vnni %s -### 2>&1 | FileCheck -check-prefix=VNNI %s
// RUN: %clang --target=i386 -march=i386 -mno-avx512vnni %s -### 2>&1 | FileCheck -check-prefix=NO-VNNI %s
// VNNI: "-target-feature" "+avx512vnni"
// NO-VNNI: "-target-feature" "-avx512vnni"

// RUN: %clang --target=i386 -march=i386 -mavx512vbmi2 %s -### 2>&1 | FileCheck -check-prefix=VBMI2 %s
// RUN: %clang --target=i386 -march=i386 -mno-avx512vbmi2 %s -### 2>&1 | FileCheck -check-prefix=NO-VBMI2 %s
// VBMI2: "-target-feature" "+avx512vbmi2"
// NO-VBMI2: "-target-feature" "-avx512vbmi2"

// RUN: %clang -target i386-linux-gnu -mavx512vp2intersect %s -### 2>&1 | FileCheck -check-prefix=VP2INTERSECT %s
// RUN: %clang -target i386-linux-gnu -mno-avx512vp2intersect %s -### 2>&1 | FileCheck -check-prefix=NO-VP2INTERSECT %s
// VP2INTERSECT: "-target-feature" "+avx512vp2intersect"
// NO-VP2INTERSECT: "-target-feature" "-avx512vp2intersect"

// RUN: %clang --target=i386 -march=i386 -mrdpid %s -### 2>&1 | FileCheck -check-prefix=RDPID %s
// RUN: %clang --target=i386 -march=i386 -mno-rdpid %s -### 2>&1 | FileCheck -check-prefix=NO-RDPID %s
// RDPID: "-target-feature" "+rdpid"
// NO-RDPID: "-target-feature" "-rdpid"

// RUN: %clang --target=i386 -march=i386 -mrdpru %s -### 2>&1 | FileCheck -check-prefix=RDPRU %s
// RUN: %clang --target=i386 -march=i386 -mno-rdpru %s -### 2>&1 | FileCheck -check-prefix=NO-RDPRU %s
// RDPRU: "-target-feature" "+rdpru"
// NO-RDPRU: "-target-feature" "-rdpru"

// RUN: %clang -target i386-linux-gnu -mretpoline %s -### 2>&1 | FileCheck -check-prefix=RETPOLINE %s
// RUN: %clang -target i386-linux-gnu -mno-retpoline %s -### 2>&1 | FileCheck -check-prefix=NO-RETPOLINE %s
// RETPOLINE: "-target-feature" "+retpoline-indirect-calls" "-target-feature" "+retpoline-indirect-branches"
// NO-RETPOLINE-NOT: retpoline

// RUN: %clang -target i386-linux-gnu -mretpoline -mretpoline-external-thunk %s -### 2>&1 | FileCheck -check-prefix=RETPOLINE-EXTERNAL-THUNK %s
// RUN: %clang -target i386-linux-gnu -mretpoline -mno-retpoline-external-thunk %s -### 2>&1 | FileCheck -check-prefix=NO-RETPOLINE-EXTERNAL-THUNK %s
// RETPOLINE-EXTERNAL-THUNK: "-target-feature" "+retpoline-external-thunk"
// NO-RETPOLINE-EXTERNAL-THUNK: "-target-feature" "-retpoline-external-thunk"

// RUN: %clang -target i386-linux-gnu -mspeculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=SLH %s
// RUN: %clang -target i386-linux-gnu -mretpoline -mspeculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=RETPOLINE %s
// RUN: %clang -target i386-linux-gnu -mno-speculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=NO-SLH %s
// SLH-NOT: retpoline
// SLH: "-target-feature" "+retpoline-indirect-calls"
// SLH-NOT: retpoline
// SLH: "-mspeculative-load-hardening"
// NO-SLH-NOT: retpoline

// RUN: %clang -target i386-linux-gnu -mlvi-cfi %s -### 2>&1 | FileCheck -check-prefix=LVICFI %s
// RUN: %clang -target i386-linux-gnu -mno-lvi-cfi %s -### 2>&1 | FileCheck -check-prefix=NO-LVICFI %s
// LVICFI: "-target-feature" "+lvi-cfi"
// NO-LVICFI-NOT: lvi-cfi

// RUN: not %clang --target=i386-linux-gnu -mlvi-cfi -mspeculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=LVICFI-SLH %s
// LVICFI-SLH: error: invalid argument 'mspeculative-load-hardening' not allowed with 'mlvi-cfi'
// RUN: not %clang --target=i386-linux-gnu -mlvi-cfi -mretpoline %s -### 2>&1 | FileCheck -check-prefix=LVICFI-RETPOLINE %s
// LVICFI-RETPOLINE: error: invalid argument 'mretpoline' not allowed with 'mlvi-cfi'
// RUN: not %clang --target=i386-linux-gnu -mlvi-cfi -mretpoline-external-thunk %s -### 2>&1 | FileCheck -check-prefix=LVICFI-RETPOLINE-EXTERNAL-THUNK %s
// LVICFI-RETPOLINE-EXTERNAL-THUNK: error: invalid argument 'mretpoline-external-thunk' not allowed with 'mlvi-cfi'

// RUN: %clang -target i386-linux-gnu -mlvi-hardening %s -### 2>&1 | FileCheck -check-prefix=LVIHARDENING %s
// RUN: %clang -target i386-linux-gnu -mno-lvi-hardening %s -### 2>&1 | FileCheck -check-prefix=NO-LVIHARDENING %s
// LVIHARDENING: "-target-feature" "+lvi-load-hardening" "-target-feature" "+lvi-cfi"
// NO-LVIHARDENING-NOT: "+lvi-

// RUN: not %clang --target=i386-linux-gnu -mlvi-hardening -mspeculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=LVIHARDENING-SLH %s
// LVIHARDENING-SLH: error: invalid argument 'mspeculative-load-hardening' not allowed with 'mlvi-hardening'
// RUN: not %clang --target=i386-linux-gnu -mlvi-hardening -mretpoline %s -### 2>&1 | FileCheck -check-prefix=LVIHARDENING-RETPOLINE %s
// LVIHARDENING-RETPOLINE: error: invalid argument 'mretpoline' not allowed with 'mlvi-hardening'
// RUN: not %clang --target=i386-linux-gnu -mlvi-hardening -mretpoline-external-thunk %s -### 2>&1 | FileCheck -check-prefix=LVIHARDENING-RETPOLINE-EXTERNAL-THUNK %s
// LVIHARDENING-RETPOLINE-EXTERNAL-THUNK: error: invalid argument 'mretpoline-external-thunk' not allowed with 'mlvi-hardening'

// RUN: %clang -target i386-linux-gnu -mseses %s -### 2>&1 | FileCheck -check-prefix=SESES %s
// RUN: %clang -target i386-linux-gnu -mno-seses %s -### 2>&1 | FileCheck -check-prefix=NO-SESES %s
// SESES: "-target-feature" "+seses"
// SESES: "-target-feature" "+lvi-cfi"
// NO-SESES-NOT: seses
// NO-SESES-NOT: lvi-cfi

// RUN: %clang -target i386-linux-gnu -mseses -mno-lvi-cfi %s -### 2>&1 | FileCheck -check-prefix=SESES-NOLVICFI %s
// SESES-NOLVICFI: "-target-feature" "+seses"
// SESES-NOLVICFI-NOT: lvi-cfi

// RUN: not %clang --target=i386-linux-gnu -mseses -mspeculative-load-hardening %s -### 2>&1 | FileCheck -check-prefix=SESES-SLH %s
// SESES-SLH: error: invalid argument 'mspeculative-load-hardening' not allowed with 'mseses'
// RUN: not %clang --target=i386-linux-gnu -mseses -mretpoline %s -### 2>&1 | FileCheck -check-prefix=SESES-RETPOLINE %s
// SESES-RETPOLINE: error: invalid argument 'mretpoline' not allowed with 'mseses'
// RUN: not %clang --target=i386-linux-gnu -mseses -mretpoline-external-thunk %s -### 2>&1 | FileCheck -check-prefix=SESES-RETPOLINE-EXTERNAL-THUNK %s
// SESES-RETPOLINE-EXTERNAL-THUNK: error: invalid argument 'mretpoline-external-thunk' not allowed with 'mseses'

// RUN: not %clang --target=i386-linux-gnu -mseses -mlvi-hardening %s -### 2>&1 | FileCheck -check-prefix=SESES-LVIHARDENING %s
// SESES-LVIHARDENING: error: invalid argument 'mlvi-hardening' not allowed with 'mseses'

// RUN: %clang -target i386-linux-gnu -mwaitpkg %s -### 2>&1 | FileCheck -check-prefix=WAITPKG %s
// RUN: %clang -target i386-linux-gnu -mno-waitpkg %s -### 2>&1 | FileCheck -check-prefix=NO-WAITPKG %s
// WAITPKG: "-target-feature" "+waitpkg"
// NO-WAITPKG: "-target-feature" "-waitpkg"

// RUN: %clang --target=i386 -march=i386 -mmovdiri %s -### 2>&1 | FileCheck -check-prefix=MOVDIRI %s
// RUN: %clang --target=i386 -march=i386 -mno-movdiri %s -### 2>&1 | FileCheck -check-prefix=NO-MOVDIRI %s
// MOVDIRI: "-target-feature" "+movdiri"
// NO-MOVDIRI: "-target-feature" "-movdiri"

// RUN: %clang --target=i386 -march=i386 -mmovdir64b %s -### 2>&1 | FileCheck -check-prefix=MOVDIR64B %s
// RUN: %clang --target=i386 -march=i386 -mno-movdir64b %s -### 2>&1 | FileCheck -check-prefix=NO-MOVDIR64B %s
// MOVDIR64B: "-target-feature" "+movdir64b"
// NO-MOVDIR64B: "-target-feature" "-movdir64b"

// RUN: %clang --target=i386 -march=i386 -mpconfig %s -### 2>&1 | FileCheck -check-prefix=PCONFIG %s
// RUN: %clang --target=i386 -march=i386 -mno-pconfig %s -### 2>&1 | FileCheck -check-prefix=NO-PCONFIG %s
// PCONFIG: "-target-feature" "+pconfig"
// NO-PCONFIG: "-target-feature" "-pconfig"

// RUN: %clang --target=i386 -march=i386 -mptwrite %s -### 2>&1 | FileCheck -check-prefix=PTWRITE %s
// RUN: %clang --target=i386 -march=i386 -mno-ptwrite %s -### 2>&1 | FileCheck -check-prefix=NO-PTWRITE %s
// PTWRITE: "-target-feature" "+ptwrite"
// NO-PTWRITE: "-target-feature" "-ptwrite"

// RUN: %clang --target=i386 -march=i386 -minvpcid %s -### 2>&1 | FileCheck -check-prefix=INVPCID %s
// RUN: %clang --target=i386 -march=i386 -mno-invpcid %s -### 2>&1 | FileCheck -check-prefix=NO-INVPCID %s
// INVPCID: "-target-feature" "+invpcid"
// NO-INVPCID: "-target-feature" "-invpcid"

// RUN: %clang --target=i386 -march=i386 -mavx512bf16 %s -### 2>&1 | FileCheck -check-prefix=AVX512BF16 %s
// RUN: %clang --target=i386 -march=i386 -mno-avx512bf16 %s -### 2>&1 | FileCheck -check-prefix=NO-AVX512BF16 %s
// AVX512BF16: "-target-feature" "+avx512bf16"
// NO-AVX512BF16: "-target-feature" "-avx512bf16"

// RUN: %clang --target=i386 -march=i386 -menqcmd %s -### 2>&1 | FileCheck --check-prefix=ENQCMD %s
// RUN: %clang --target=i386 -march=i386 -mno-enqcmd %s -### 2>&1 | FileCheck --check-prefix=NO-ENQCMD %s
// ENQCMD: "-target-feature" "+enqcmd"
// NO-ENQCMD: "-target-feature" "-enqcmd"

// RUN: %clang --target=i386 -march=i386 -mvzeroupper %s -### 2>&1 | FileCheck --check-prefix=VZEROUPPER %s
// RUN: %clang --target=i386 -march=i386 -mno-vzeroupper %s -### 2>&1 | FileCheck --check-prefix=NO-VZEROUPPER %s
// VZEROUPPER: "-target-feature" "+vzeroupper"
// NO-VZEROUPPER: "-target-feature" "-vzeroupper"

// RUN: %clang --target=i386 -march=i386 -mserialize %s -### 2>&1 | FileCheck -check-prefix=SERIALIZE %s
// RUN: %clang --target=i386 -march=i386 -mno-serialize %s -### 2>&1 | FileCheck -check-prefix=NO-SERIALIZE %s
// SERIALIZE: "-target-feature" "+serialize"
// NO-SERIALIZE: "-target-feature" "-serialize"

// RUN: %clang --target=i386 -march=i386 -mtsxldtrk %s -### 2>&1 | FileCheck --check-prefix=TSXLDTRK %s
// RUN: %clang --target=i386 -march=i386 -mno-tsxldtrk %s -### 2>&1 | FileCheck --check-prefix=NO-TSXLDTRK %s
// TSXLDTRK: "-target-feature" "+tsxldtrk"
// NO-TSXLDTRK: "-target-feature" "-tsxldtrk"

// RUN: %clang -target i386-linux-gnu -mkl %s -### 2>&1 | FileCheck -check-prefix=KL %s
// RUN: %clang -target i386-linux-gnu -mno-kl %s -### 2>&1 | FileCheck -check-prefix=NO-KL %s
// KL: "-target-feature" "+kl"
// NO-KL: "-target-feature" "-kl"

// RUN: %clang -target i386-linux-gnu -mwidekl %s -### 2>&1 | FileCheck -check-prefix=WIDE_KL %s
// RUN: %clang -target i386-linux-gnu -mno-widekl %s -### 2>&1 | FileCheck -check-prefix=NO-WIDE_KL %s
// WIDE_KL: "-target-feature" "+widekl"
// NO-WIDE_KL: "-target-feature" "-widekl"

// RUN: %clang --target=i386 -march=i386 -mamx-tile %s -### 2>&1 | FileCheck --check-prefix=AMX-TILE %s
// RUN: %clang --target=i386 -march=i386 -mno-amx-tile %s -### 2>&1 | FileCheck --check-prefix=NO-AMX-TILE %s
// AMX-TILE: "-target-feature" "+amx-tile"
// NO-AMX-TILE: "-target-feature" "-amx-tile"

// RUN: %clang --target=i386 -march=i386 -mamx-bf16 %s -### 2>&1 | FileCheck --check-prefix=AMX-BF16 %s
// RUN: %clang --target=i386 -march=i386 -mno-amx-bf16 %s -### 2>&1 | FileCheck -check-prefix=NO-AMX-BF16 %s
// AMX-BF16: "-target-feature" "+amx-bf16"
// NO-AMX-BF16: "-target-feature" "-amx-bf16"

// RUN: %clang --target=i386 -march=i386 -mamx-int8 %s -### 2>&1 | FileCheck --check-prefix=AMX-INT8 %s
// RUN: %clang --target=i386 -march=i386 -mno-amx-int8 %s -### 2>&1 | FileCheck --check-prefix=NO-AMX-INT8 %s
// AMX-INT8: "-target-feature" "+amx-int8"
// NO-AMX-INT8: "-target-feature" "-amx-int8"

// RUN: %clang --target=x86_64 -mamx-fp16 %s \
// RUN: -### -o %t.o 2>&1 | FileCheck -check-prefix=AMX-FP16 %s
// RUN: %clang --target=x86_64 -mno-amx-fp16 \
// RUN: %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AMX-FP16 %s
// AMX-FP16: "-target-feature" "+amx-fp16"
// NO-AMX-FP16: "-target-feature" "-amx-fp16"

// RUN: %clang -target x86_64-unknown-linux-gnu -mamx-complex %s \
// RUN: -### -o %t.o 2>&1 | FileCheck -check-prefix=AMX-COMPLEX %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-amx-complex %s \
// RUN: -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AMX-COMPLEX %s
// AMX-COMPLEX: "-target-feature" "+amx-complex"
// NO-AMX-COMPLEX: "-target-feature" "-amx-complex"

// RUN: %clang --target=i386 -march=i386 -mhreset %s -### 2>&1 | FileCheck -check-prefix=HRESET %s
// RUN: %clang --target=i386 -march=i386 -mno-hreset %s -### 2>&1 | FileCheck -check-prefix=NO-HRESET %s
// HRESET: "-target-feature" "+hreset"
// NO-HRESET: "-target-feature" "-hreset"

// RUN: %clang --target=i386 -march=i386 -muintr %s -### 2>&1 | FileCheck -check-prefix=UINTR %s
// RUN: %clang --target=i386 -march=i386 -mno-uintr %s -### 2>&1 | FileCheck -check-prefix=NO-UINTR %s
// UINTR: "-target-feature" "+uintr"
// NO-UINTR: "-target-feature" "-uintr"

// RUN: %clang --target=i386 -march=i386 -mavxvnni %s -### 2>&1 | FileCheck --check-prefix=AVX-VNNI %s
// RUN: %clang --target=i386 -march=i386 -mno-avxvnni %s -### 2>&1 | FileCheck --check-prefix=NO-AVX-VNNI %s
// AVX-VNNI: "-target-feature" "+avxvnni"
// NO-AVX-VNNI: "-target-feature" "-avxvnni"

// RUN: %clang --target=i386 -march=i386 -mavx512fp16 %s -### 2>&1 | FileCheck -check-prefix=AVX512FP16 %s
// RUN: %clang --target=i386 -march=i386 -mno-avx512fp16 %s -### 2>&1 | FileCheck -check-prefix=NO-AVX512FP16 %s
// AVX512FP16: "-target-feature" "+avx512fp16"
// NO-AVX512FP16: "-target-feature" "-avx512fp16"

// RUN: %clang --target=x86_64 -mcmpccxadd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CMPCCXADD %s
// RUN: %clang --target=x86_64 -mno-cmpccxadd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-CMPCCXADD %s
// CMPCCXADD: "-target-feature" "+cmpccxadd"
// NO-CMPCCXADD: "-target-feature" "-cmpccxadd"

// RUN: %clang --target=i386 -march=i386 -mraoint %s -### 2>&1 | FileCheck -check-prefix=RAOINT %s
// RUN: %clang --target=i386 -march=i386 -mno-raoint %s -### 2>&1 | FileCheck -check-prefix=NO-RAOINT %s
// RAOINT: "-target-feature" "+raoint"
// NO-RAOINT: "-target-feature" "-raoint"

// RUN: %clang -target i386-linux-gnu -mavxifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVXIFMA %s
// RUN: %clang -target i386-linux-gnu -mno-avxifma %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVXIFMA %s
// AVXIFMA: "-target-feature" "+avxifma"
// NO-AVXIFMA: "-target-feature" "-avxifma"

// RUN: %clang --target=i386 -mavxvnniint8 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX-VNNIINT8 %s
// RUN: %clang --target=i386 -mno-avxvnniint8 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVX-VNNIINT8 %s
// AVX-VNNIINT8: "-target-feature" "+avxvnniint8"
// NO-AVX-VNNIINT8: "-target-feature" "-avxvnniint8"

// RUN: %clang --target=i386 -mavxneconvert %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVXNECONVERT %s
// RUN: %clang --target=i386 -mno-avxneconvert %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVXNECONVERT %s
// AVXNECONVERT: "-target-feature" "+avxneconvert"
// NO-AVXNECONVERT: "-target-feature" "-avxneconvert"

// RUN: %clang --target=i386 -msha512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SHA512 %s
// RUN: %clang --target=i386 -mno-sha512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SHA512 %s
// SHA512: "-target-feature" "+sha512"
// NO-SHA512: "-target-feature" "-sha512"

// RUN: %clang --target=i386 -msm3 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SM3 %s
// RUN: %clang --target=i386 -mno-sm3 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SM3 %s
// SM3: "-target-feature" "+sm3"
// NO-SM3: "-target-feature" "-sm3"

// RUN: %clang --target=i386 -msm4 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=SM4 %s
// RUN: %clang --target=i386 -mno-sm4 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SM4 %s
// SM4: "-target-feature" "+sm4"
// NO-SM4: "-target-feature" "-sm4"

// RUN: %clang --target=i386 -mavxvnniint16 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVXVNNIINT16 %s
// RUN: %clang --target=i386 -mno-avxvnniint16 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-AVXVNNIINT16 %s
// AVXVNNIINT16: "-target-feature" "+avxvnniint16"
// NO-AVXVNNIINT16: "-target-feature" "-avxvnniint16"

// RUN: %clang --target=i386 -mevex512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EVEX512 %s
// RUN: %clang --target=i386 -mno-evex512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EVEX512 %s
// EVEX512: "-target-feature" "+evex512"
// NO-EVEX512: "-target-feature" "-evex512"

// RUN: %clang --target=i386 -mavx10.1 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10_1_256 %s
// RUN: %clang --target=i386 -mavx10.1-256 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10_1_256 %s
// RUN: %clang --target=i386 -mavx10.1-512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10_1_512 %s
// RUN: %clang --target=i386 -mavx10.1-256 -mavx10.1-512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10_1_512 %s
// RUN: %clang --target=i386 -mavx10.1-512 -mavx10.1-256 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10_1_256 %s
// RUN: not %clang --target=i386 -march=i386 -mavx10.1-128 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BAD-AVX10 %s
// RUN: not %clang --target=i386 -march=i386 -mavx10.a-256 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BAD-AVX10 %s
// RUN: not %clang --target=i386 -march=i386 -mavx10.1024-512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BAD-AVX10 %s
// RUN: %clang --target=i386 -march=i386 -mavx10.1 -mavx512f %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10-AVX512 %s
// RUN: %clang --target=i386 -march=i386 -mavx10.1 -mno-avx512f %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10-AVX512 %s
// RUN: %clang --target=i386 -march=i386 -mavx10.1 -mevex512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10-EVEX512 %s
// RUN: %clang --target=i386 -march=i386 -mavx10.1 -mno-evex512 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=AVX10-EVEX512 %s
// AVX10_1_256: "-target-feature" "+avx10.1-256"
// AVX10_1_512: "-target-feature" "+avx10.1-512"
// BAD-AVX10: error: unknown argument{{:?}} '-mavx10.{{.*}}'
// AVX10-AVX512: "-target-feature" "+avx10.1-256" "-target-feature" "{{.}}avx512f"
// AVX10-EVEX512: "-target-feature" "+avx10.1-256" "-target-feature" "{{.}}evex512"

// RUN: %clang --target=i386 -musermsr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=USERMSR %s
// RUN: %clang --target=i386 -mno-usermsr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-USERMSR %s
// USERMSR: "-target-feature" "+usermsr"
// NO-USERMSR: "-target-feature" "-usermsr"

// RUN: %clang --target=i386 -march=i386 -mcrc32 %s -### 2>&1 | FileCheck -check-prefix=CRC32 %s
// RUN: %clang --target=i386 -march=i386 -mno-crc32 %s -### 2>&1 | FileCheck -check-prefix=NO-CRC32 %s
// CRC32: "-target-feature" "+crc32"
// NO-CRC32: "-target-feature" "-crc32"

// RUN: not %clang -### --target=aarch64 -mcrc32 -msse4.1 -msse4.2 -mno-sgx %s 2>&1 | FileCheck --check-prefix=NONX86 %s
// NONX86:      error: unsupported option '-mcrc32' for target 'aarch64'
// NONX86-NEXT: error: unsupported option '-msse4.1' for target 'aarch64'
/// TODO: This warning is a workaround for https://github.com/llvm/llvm-project/issues/63270
// NONX86-NEXT: warning: argument unused during compilation: '-msse4.2' [-Wunused-command-line-argument]
// NONX86-NEXT: error: unsupported option '-mno-sgx' for target 'aarch64'

// RUN: %clang --target=i386 -march=i386 -mharden-sls=return %s -### -o %t.o 2>&1 | FileCheck -check-prefixes=SLS-RET,NO-SLS %s
// RUN: %clang --target=i386 -march=i386 -mharden-sls=indirect-jmp %s -### -o %t.o 2>&1 | FileCheck -check-prefixes=SLS-IJMP,NO-SLS %s
// RUN: %clang --target=i386 -march=i386 -mharden-sls=none -mharden-sls=all %s -### -o %t.o 2>&1 | FileCheck -check-prefixes=SLS-IJMP,SLS-RET %s
// RUN: %clang --target=i386 -march=i386 -mharden-sls=all -mharden-sls=none %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-SLS %s
// RUN: not %clang --target=i386 -march=i386 -mharden-sls=return,indirect-jmp %s -### -o %t.o 2>&1 | FileCheck -check-prefix=BAD-SLS %s
// NO-SLS-NOT: "+harden-sls-
// SLS-RET-DAG: "-target-feature" "+harden-sls-ret"
// SLS-IJMP-DAG: "-target-feature" "+harden-sls-ijmp"
// NO-SLS-NOT: "+harden-sls-
// BAD-SLS: unsupported argument '{{[^']+}}' to option '-mharden-sls='

// RUN: touch %t.o
// RUN: %clang -fdriver-only -Werror --target=x86_64-pc-linux-gnu -mharden-sls=all %t.o -o /dev/null 2>&1 | count 0

// RUN: %clang -target x86_64-unknown-linux-gnu -mapxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apxf -mapxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=APXF %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapxf -mno-apxf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-APXF %s
//
// APXF: "-target-feature" "+egpr" "-target-feature" "+push2pop2" "-target-feature" "+ppx"
// NO-APXF: "-target-feature" "-egpr" "-target-feature" "-push2pop2" "-target-feature" "-ppx"

// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=push2pop2 %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PUSH2POP2 %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=ppx %s -### -o %t.o 2>&1 | FileCheck -check-prefix=PPX %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=ccmp %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CCMP %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=cf %s -### -o %t.o 2>&1 | FileCheck -check-prefix=CF %s
// EGPR: "-target-feature" "+egpr"
// PUSH2POP2: "-target-feature" "+push2pop2"
// PPX: "-target-feature" "+ppx"
// NDD: "-target-feature" "+ndd"
// CCMP: "-target-feature" "+ccmp"
// CF: "-target-feature" "+cf"

// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr -mapx-features=ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr -mno-apx-features=ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NO-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr -mapx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mno-apx-features=egpr,ndd -mapx-features=egpr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=EGPR-NO-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,ndd -mno-apx-features=egpr %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NDD %s
// RUN: %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr -mno-apx-features=egpr,ndd %s -### -o %t.o 2>&1 | FileCheck -check-prefix=NO-EGPR-NO-NDD %s
//
// EGPR-NDD: "-target-feature" "+egpr" "-target-feature" "+ndd"
// EGPR-NO-NDD: "-target-feature" "-ndd" "-target-feature" "+egpr"
// NO-EGPR-NDD: "-target-feature" "+ndd" "-target-feature" "-egpr"
// NO-EGPR-NO-NDD: "-target-feature" "-egpr" "-target-feature" "-ndd"

// RUN: not %clang -target x86_64-unknown-linux-gnu -mapx-features=egpr,foo,bar %s -### -o %t.o 2>&1 | FileCheck -check-prefix=ERROR %s
//
// ERROR: unsupported argument 'foo' to option '-mapx-features='
// ERROR: unsupported argument 'bar' to option '-mapx-features='
