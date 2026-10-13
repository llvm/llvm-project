! Test that -mcpu/march are used and that the -target-cpu and -target-features
! are also added to the fc1 command.
!
! The X86 section batches m_x86_Features_Group options by ISA family (AMX, SSE,
! AVX/AVX512, then other). Each family has a matching -mno-* RUN.

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A57

! RUN: %flang --target=aarch64-linux-gnu -mcpu=cortex-a76 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-A76

! RUN: %flang --target=aarch64-linux-gnu -march=armv9 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-ARMV9

! Negative test. ARM cpu with x86 target.
! RUN: not %flang --target=x86_64-linux-gnu -mcpu=cortex-a57 -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-A57

! RUN: %flang --target=x86_64-linux-gnu -march=skylake -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-SKYLAKE

! RUN: %flang --target=x86_64-linux-gnu -mapx-features=egpr -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-APX

! RUN: %flang --target=x86_64-linux-gnu -mno-apx-features=ccmp -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-NO-APX

! RUN: %flang --target=x86_64h-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-X86_64H

! RUN: %flang --target=riscv64-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-RV64

! RUN: %flang --target=amdgcn-amd-amdhsa -mcpu=gfx908 -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-AMDGPU

! RUN: %flang --target=r600-unknown-unknown -mcpu=cayman -nogpulib -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-AMDGPU-R600

! RUN: %flang --target=loongarch64-linux-gnu -c %s -### 2>&1 \
! RUN: | FileCheck %s -check-prefix=CHECK-LOONGARCH64

! RUN: %flang --target=sparc64-linux-gnu -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS
! RUN: %flang --target=sparc64-freebsd -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS
! RUN: %flang --target=sparc64-openbsd -c -### %s 2>&1  | FileCheck %s -check-prefix=CHECK-SPARC-VIS

! CHECK-A57: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A57-SAME: "-target-cpu" "cortex-a57"
! CHECK-A57-SAME: "-target-feature" "+v8a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+fp-armv8" "-target-feature" "+neon" "-target-feature" "+perfmon" "-target-feature" "+sha2

! CHECK-A76: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-A76-SAME: "-target-cpu" "cortex-a76"
! CHECK-A76-SAME: "-target-feature" "+v8.2a" "-target-feature" "+aes" "-target-feature" "+crc" "-target-feature" "+dotprod" "-target-feature" "+fp-armv8" "-target-feature" "+fullfp16" "-target-feature" "+lse" "-target-feature" "+neon" "-target-feature" "+perfmon" "-target-feature" "+ras" "-target-feature" "+rcpc" "-target-feature" "+rdm" "-target-feature" "+sha2" "-target-feature" "+ssbs"

! CHECK-ARMV9: "-fc1" "-triple" "aarch64-unknown-linux-gnu"
! CHECK-ARMV9-SAME: "-target-cpu" "generic"
! CHECK-ARMV9-SAME: "-target-feature" "+v9a"
! CHECK-ARMV9-SAME: "-target-feature" "+sve"
! CHECK-ARMV9-SAME: "-target-feature" "+sve2"

! CHECK-NO-A57: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-A57-NOT: cortex-a57
! CHECK-NO-A57-SAME: "-target-cpu" "x86-64"
! CHECK-NO-A57-NOT: cortex-a57

! CHECK-SKYLAKE: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-SKYLAKE-SAME: "-target-cpu" "skylake"

! CHECK-APX: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-APX-SAME: "-target-feature" "+egpr"

! CHECK-NO-APX: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-NO-APX-SAME: "-target-feature" "-ccmp"

! CHECK-X86_64H: "-fc1" "-triple" "x86_64h-unknown-linux-gnu"
! CHECK-X86_64H-SAME: "-target-cpu" "x86-64" "-target-feature" "-rdrnd" "-target-feature" "-aes" "-target-feature" "-pclmul" "-target-feature" "-rtm" "-target-feature" "-fsgsbase"

! CHECK-RV64: "-fc1" "-triple" "riscv64-unknown-linux-gnu"
! CHECK-RV64-SAME: "-target-cpu" "generic-rv64" "-target-feature" "+i" "-target-feature" "+m" "-target-feature" "+a" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+c"

! CHECK-AMDGPU: "-fc1" "-triple" "amdgpu9.08-amd-amdhsa"
! CHECK-AMDGPU-SAME: "-target-cpu" "gfx908"

! CHECK-AMDGPU-R600: "-fc1" "-triple" "r600-unknown-unknown"
! CHECK-AMDGPU-R600-SAME: "-target-cpu" "cayman"

! CHECK-LOONGARCH64: "-fc1" "-triple" "loongarch64-unknown-linux-gnu"
! CHECK-LOONGARCH64-SAME: "-target-cpu" "loongarch64" "-target-feature" "+lsx" "-target-feature" "+relax" "-target-feature" "+64bit" "-target-feature" "+f" "-target-feature" "+d" "-target-feature" "+ual"

! CHECK-SPARC-VIS: "-fc1" "-triple" "sparc64-{{[^"]+}}"
! CHECK-SPARC-VIS-SAME: "-target-feature" "+vis"

! AMX
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mamx-avx512 -mamx-bf16 -mamx-complex -mamx-fp16 -mamx-int8 -mamx-fp8 -mamx-tile -mamx-movrs \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-AMX
! CHECK-X86-AMX: "-fc1"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-avx512"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-bf16"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-complex"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-fp16"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-int8"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-fp8"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-tile"
! CHECK-X86-AMX-SAME: "-target-feature" "+amx-movrs"
!
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mno-amx-avx512 -mno-amx-bf16 -mno-amx-complex -mno-amx-fp16 -mno-amx-int8 -mno-amx-fp8 -mno-amx-tile -mno-amx-movrs \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-AMX-NEG
! CHECK-X86-AMX-NEG: "-fc1"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-avx512"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-bf16"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-complex"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-fp16"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-int8"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-fp8"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-tile"
! CHECK-X86-AMX-NEG-SAME: "-target-feature" "-amx-movrs"
!
! SSE
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -msse4a \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-SSE
! CHECK-X86-SSE: "-fc1"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse2"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse3"
! CHECK-X86-SSE-SAME: "-target-feature" "+ssse3"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse4.1"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse4.2"
! CHECK-X86-SSE-SAME: "-target-feature" "+sse4a"
!
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mno-sse -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -mno-sse4a \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-SSE-NEG
! CHECK-X86-SSE-NEG: "-fc1"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse2"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse3"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-ssse3"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse4.1"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse4.2"
! CHECK-X86-SSE-NEG-SAME: "-target-feature" "-sse4a"
!
! AVX (including AVX10 / AVX512 / AVX-VNNI)
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mavx -mavx10.1 -mavx10.2 -mavx2 \
! RUN:   -mavx512f -mavx512bf16 -mavx512bitalg -mavx512bmm -mavx512bw -mavx512cd -mavx512dq -mavx512fp16 \
! RUN:   -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512vnni -mavx512vpopcntdq -mavx512vp2intersect \
! RUN:   -mavxifma -mavxneconvert -mavxvnniint16 -mavxvnniint8 -mavxvnni \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-AVX
! CHECK-X86-AVX: "-fc1"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx10.1"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx10.2"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx2"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512f"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512bf16"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512bitalg"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512bmm"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512bw"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512cd"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512dq"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512fp16"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512ifma"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vbmi"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vbmi2"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vl"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vnni"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vpopcntdq"
! CHECK-X86-AVX-SAME: "-target-feature" "+avx512vp2intersect"
! CHECK-X86-AVX-SAME: "-target-feature" "+avxifma"
! CHECK-X86-AVX-SAME: "-target-feature" "+avxneconvert"
! CHECK-X86-AVX-SAME: "-target-feature" "+avxvnniint16"
! CHECK-X86-AVX-SAME: "-target-feature" "+avxvnniint8"
! CHECK-X86-AVX-SAME: "-target-feature" "+avxvnni"
!
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mno-avx -mno-avx10.1 -mno-avx10.2 -mno-avx2 \
! RUN:   -mno-avx512f -mno-avx512bf16 -mno-avx512bitalg -mno-avx512bmm -mno-avx512bw -mno-avx512cd -mno-avx512dq -mno-avx512fp16 \
! RUN:   -mno-avx512ifma -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512vl -mno-avx512vnni -mno-avx512vpopcntdq -mno-avx512vp2intersect \
! RUN:   -mno-avxifma -mno-avxneconvert -mno-avxvnniint16 -mno-avxvnniint8 -mno-avxvnni \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-AVX-NEG
! CHECK-X86-AVX-NEG: "-fc1"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx10.1"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx10.2"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx2"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512f"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512bf16"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512bitalg"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512bmm"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512bw"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512cd"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512dq"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512fp16"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512ifma"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vbmi"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vbmi2"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vl"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vnni"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vpopcntdq"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avx512vp2intersect"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avxifma"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avxneconvert"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avxvnniint16"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avxvnniint8"
! CHECK-X86-AVX-NEG-SAME: "-target-feature" "-avxvnni"
!
! Remaining m_x86_Features_Group flags.
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mx87 -mmmx -mcmpccxadd \
! RUN:   -madx -maes -mbmi -mbmi2 -mcldemote -mclflushopt -mclwb -mwbnoinvd -mclzero -mcrc32 -mcx16 -menqcmd -mf16c -mfma -mfma4 -mfsgsbase -mfxsr -minvpcid \
! RUN:   -mgfni -mhreset -mkl -mwidekl -mlwp -mlzcnt -mmovbe -mmovdiri -mmovdir64b -mmovrs -mmwaitx -mpku -mpclmul -mpconfig -mpopcnt -mprefetchi -mprfchw -mptwrite -mraoint -mrdpid -mrdpru -mrdrnd -mrtm -mrdseed -msahf -mserialize -msgx -msha -msha512 -msm3 \
! RUN:   -msm4 -mtbm -mtsxldtrk -muintr -musermsr -mvaes -mvpclmulqdq -mwaitpkg -mxop -mxsave -mxsavec -mxsaveopt -mxsaves -mshstk -mretpoline-external-thunk -mvzeroupper \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-OTHER
! CHECK-X86-OTHER: "-fc1"
! CHECK-X86-OTHER-SAME: "-target-feature" "+x87"
! CHECK-X86-OTHER-SAME: "-target-feature" "+mmx"
! CHECK-X86-OTHER-SAME: "-target-feature" "+cmpccxadd"
! CHECK-X86-OTHER-SAME: "-target-feature" "+adx"
! CHECK-X86-OTHER-SAME: "-target-feature" "+aes"
! CHECK-X86-OTHER-SAME: "-target-feature" "+bmi"
! CHECK-X86-OTHER-SAME: "-target-feature" "+bmi2"
! CHECK-X86-OTHER-SAME: "-target-feature" "+cldemote"
! CHECK-X86-OTHER-SAME: "-target-feature" "+clflushopt"
! CHECK-X86-OTHER-SAME: "-target-feature" "+clwb"
! CHECK-X86-OTHER-SAME: "-target-feature" "+wbnoinvd"
! CHECK-X86-OTHER-SAME: "-target-feature" "+clzero"
! CHECK-X86-OTHER-SAME: "-target-feature" "+crc32"
! CHECK-X86-OTHER-SAME: "-target-feature" "+cx16"
! CHECK-X86-OTHER-SAME: "-target-feature" "+enqcmd"
! CHECK-X86-OTHER-SAME: "-target-feature" "+f16c"
! CHECK-X86-OTHER-SAME: "-target-feature" "+fma"
! CHECK-X86-OTHER-SAME: "-target-feature" "+fma4"
! CHECK-X86-OTHER-SAME: "-target-feature" "+fsgsbase"
! CHECK-X86-OTHER-SAME: "-target-feature" "+fxsr"
! CHECK-X86-OTHER-SAME: "-target-feature" "+invpcid"
! CHECK-X86-OTHER-SAME: "-target-feature" "+gfni"
! CHECK-X86-OTHER-SAME: "-target-feature" "+hreset"
! CHECK-X86-OTHER-SAME: "-target-feature" "+kl"
! CHECK-X86-OTHER-SAME: "-target-feature" "+widekl"
! CHECK-X86-OTHER-SAME: "-target-feature" "+lwp"
! CHECK-X86-OTHER-SAME: "-target-feature" "+lzcnt"
! CHECK-X86-OTHER-SAME: "-target-feature" "+movbe"
! CHECK-X86-OTHER-SAME: "-target-feature" "+movdiri"
! CHECK-X86-OTHER-SAME: "-target-feature" "+movdir64b"
! CHECK-X86-OTHER-SAME: "-target-feature" "+movrs"
! CHECK-X86-OTHER-SAME: "-target-feature" "+mwaitx"
! CHECK-X86-OTHER-SAME: "-target-feature" "+pku"
! CHECK-X86-OTHER-SAME: "-target-feature" "+pclmul"
! CHECK-X86-OTHER-SAME: "-target-feature" "+pconfig"
! CHECK-X86-OTHER-SAME: "-target-feature" "+popcnt"
! CHECK-X86-OTHER-SAME: "-target-feature" "+prefetchi"
! CHECK-X86-OTHER-SAME: "-target-feature" "+prfchw"
! CHECK-X86-OTHER-SAME: "-target-feature" "+ptwrite"
! CHECK-X86-OTHER-SAME: "-target-feature" "+raoint"
! CHECK-X86-OTHER-SAME: "-target-feature" "+rdpid"
! CHECK-X86-OTHER-SAME: "-target-feature" "+rdpru"
! CHECK-X86-OTHER-SAME: "-target-feature" "+rdrnd"
! CHECK-X86-OTHER-SAME: "-target-feature" "+rtm"
! CHECK-X86-OTHER-SAME: "-target-feature" "+rdseed"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sahf"
! CHECK-X86-OTHER-SAME: "-target-feature" "+serialize"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sgx"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sha"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sha512"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sm3"
! CHECK-X86-OTHER-SAME: "-target-feature" "+sm4"
! CHECK-X86-OTHER-SAME: "-target-feature" "+tbm"
! CHECK-X86-OTHER-SAME: "-target-feature" "+tsxldtrk"
! CHECK-X86-OTHER-SAME: "-target-feature" "+uintr"
! CHECK-X86-OTHER-SAME: "-target-feature" "+usermsr"
! CHECK-X86-OTHER-SAME: "-target-feature" "+vaes"
! CHECK-X86-OTHER-SAME: "-target-feature" "+vpclmulqdq"
! CHECK-X86-OTHER-SAME: "-target-feature" "+waitpkg"
! CHECK-X86-OTHER-SAME: "-target-feature" "+xop"
! CHECK-X86-OTHER-SAME: "-target-feature" "+xsave"
! CHECK-X86-OTHER-SAME: "-target-feature" "+xsavec"
! CHECK-X86-OTHER-SAME: "-target-feature" "+xsaveopt"
! CHECK-X86-OTHER-SAME: "-target-feature" "+xsaves"
! CHECK-X86-OTHER-SAME: "-target-feature" "+shstk"
! CHECK-X86-OTHER-SAME: "-target-feature" "+retpoline-external-thunk"
! CHECK-X86-OTHER-SAME: "-target-feature" "+vzeroupper"
!
! RUN: %flang --target=x86_64-linux-gnu \
! RUN:   -mno-x87 -mno-mmx -mno-cmpccxadd \
! RUN:   -mno-adx -mno-aes -mno-bmi -mno-bmi2 -mno-cldemote -mno-clflushopt -mno-clwb -mno-wbnoinvd -mno-clzero -mno-crc32 -mno-cx16 -mno-enqcmd -mno-f16c -mno-fma -mno-fma4 -mno-fsgsbase -mno-fxsr -mno-invpcid \
! RUN:   -mno-gfni -mno-hreset -mno-kl -mno-widekl -mno-lwp -mno-lzcnt -mno-movbe -mno-movdiri -mno-movdir64b -mno-movrs -mno-mwaitx -mno-pku -mno-pclmul -mno-pconfig -mno-popcnt -mno-prefetchi -mno-prfchw -mno-ptwrite -mno-raoint -mno-rdpid -mno-rdpru -mno-rdrnd -mno-rtm -mno-rdseed -mno-sahf -mno-serialize -mno-sgx -mno-sha -mno-sha512 -mno-sm3 \
! RUN:   -mno-sm4 -mno-tbm -mno-tsxldtrk -mno-uintr -mno-usermsr -mno-vaes -mno-vpclmulqdq -mno-waitpkg -mno-xop -mno-xsave -mno-xsavec -mno-xsaveopt -mno-xsaves -mno-shstk -mno-retpoline-external-thunk -mno-vzeroupper \
! RUN:   %s -### 2>&1 | FileCheck %s -check-prefix=CHECK-X86-OTHER-NEG
! CHECK-X86-OTHER-NEG: "-fc1"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-x87"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-mmx"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-cmpccxadd"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-adx"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-aes"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-bmi"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-bmi2"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-cldemote"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-clflushopt"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-clwb"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-wbnoinvd"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-clzero"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-crc32"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-cx16"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-enqcmd"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-f16c"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-fma"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-fma4"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-fsgsbase"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-fxsr"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-invpcid"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-gfni"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-hreset"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-kl"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-widekl"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-lwp"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-lzcnt"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-movbe"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-movdiri"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-movdir64b"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-movrs"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-mwaitx"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-pku"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-pclmul"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-pconfig"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-popcnt"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-prefetchi"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-prfchw"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-ptwrite"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-raoint"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-rdpid"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-rdpru"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-rdrnd"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-rtm"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-rdseed"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sahf"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-serialize"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sgx"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sha"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sha512"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sm3"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-sm4"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-tbm"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-tsxldtrk"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-uintr"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-usermsr"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-vaes"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-vpclmulqdq"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-waitpkg"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-xop"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-xsave"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-xsavec"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-xsaveopt"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-xsaves"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-shstk"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-retpoline-external-thunk"
! CHECK-X86-OTHER-NEG-SAME: "-target-feature" "-vzeroupper"
!
! RUN: %flang --target=x86_64-linux-gnu -mno-gather -mno-scatter %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-GATHER-SCATTER
! CHECK-X86-GATHER-SCATTER: "-target-feature" "+prefer-no-gather"
! CHECK-X86-GATHER-SCATTER-SAME: "-target-feature" "+prefer-no-scatter"

! RUN: not %flang --target=aarch64-linux-gnu -mavx -mcrc32 %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-NONX86
! CHECK-NONX86: error: unsupported option '-mavx' for target 'aarch64-linux-gnu'
! CHECK-NONX86: error: unsupported option '-mcrc32' for target 'aarch64-linux-gnu'

