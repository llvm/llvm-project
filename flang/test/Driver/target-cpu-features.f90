! Test that -mcpu/march are used and that the -target-cpu and -target-features
! are also added to the fc1 command.
!
! The X86 section uses grouped RUN lines to verify m_x86_Features_Group options
! are accepted and forwarded as -target-feature to flang -fc1.

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

!
! ============================================================================
! X86 m_x86_Features_Group: grouped checks to minimize driver invocations.
! ============================================================================
!
! RUN: %flang --target=x86_64-linux-gnu -mx87 -mmmx -mamx-avx512 -mamx-bf16 -mamx-complex -mamx-fp16 -mamx-int8 -mamx-fp8 -mamx-tile -mamx-movrs -mcmpccxadd -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -msse4a -mavx -mavx10.1 -mavx10.2 -mavx2 -mavx512f -mavx512bf16 -mavx512bitalg -mavx512bmm -mavx512bw -mavx512cd -mavx512dq -mavx512fp16 -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-G1
! CHECK-X86-G1: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-G1-SAME: "-target-feature" "+x87"
! CHECK-X86-G1-SAME: "-target-feature" "+mmx"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-avx512"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-bf16"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-complex"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-fp16"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-int8"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-fp8"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-tile"
! CHECK-X86-G1-SAME: "-target-feature" "+amx-movrs"
! CHECK-X86-G1-SAME: "-target-feature" "+cmpccxadd"
! CHECK-X86-G1-SAME: "-target-feature" "+sse"
! CHECK-X86-G1-SAME: "-target-feature" "+sse2"
! CHECK-X86-G1-SAME: "-target-feature" "+sse3"
! CHECK-X86-G1-SAME: "-target-feature" "+ssse3"
! CHECK-X86-G1-SAME: "-target-feature" "+sse4.1"
! CHECK-X86-G1-SAME: "-target-feature" "+sse4.2"
! CHECK-X86-G1-SAME: "-target-feature" "+sse4a"
! CHECK-X86-G1-SAME: "-target-feature" "+avx"
! CHECK-X86-G1-SAME: "-target-feature" "+avx10.1"
! CHECK-X86-G1-SAME: "-target-feature" "+avx10.2"
! CHECK-X86-G1-SAME: "-target-feature" "+avx2"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512f"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512bf16"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512bitalg"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512bmm"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512bw"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512cd"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512dq"
! CHECK-X86-G1-SAME: "-target-feature" "+avx512fp16"
!
! RUN: %flang --target=x86_64-linux-gnu -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512vnni -mavx512vpopcntdq -mavx512vp2intersect -mavxifma -mavxneconvert -mavxvnniint16 -mavxvnniint8 -mavxvnni -madx -maes -mbmi -mbmi2 -mcldemote -mclflushopt -mclwb -mwbnoinvd -mclzero -mcrc32 -mcx16 -menqcmd -mf16c -mfma -mfma4 -mfsgsbase -mfxsr -minvpcid -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-G2
! CHECK-X86-G2: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512ifma"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vbmi"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vbmi2"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vl"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vnni"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vpopcntdq"
! CHECK-X86-G2-SAME: "-target-feature" "+avx512vp2intersect"
! CHECK-X86-G2-SAME: "-target-feature" "+avxifma"
! CHECK-X86-G2-SAME: "-target-feature" "+avxneconvert"
! CHECK-X86-G2-SAME: "-target-feature" "+avxvnniint16"
! CHECK-X86-G2-SAME: "-target-feature" "+avxvnniint8"
! CHECK-X86-G2-SAME: "-target-feature" "+avxvnni"
! CHECK-X86-G2-SAME: "-target-feature" "+adx"
! CHECK-X86-G2-SAME: "-target-feature" "+aes"
! CHECK-X86-G2-SAME: "-target-feature" "+bmi"
! CHECK-X86-G2-SAME: "-target-feature" "+bmi2"
! CHECK-X86-G2-SAME: "-target-feature" "+cldemote"
! CHECK-X86-G2-SAME: "-target-feature" "+clflushopt"
! CHECK-X86-G2-SAME: "-target-feature" "+clwb"
! CHECK-X86-G2-SAME: "-target-feature" "+wbnoinvd"
! CHECK-X86-G2-SAME: "-target-feature" "+clzero"
! CHECK-X86-G2-SAME: "-target-feature" "+crc32"
! CHECK-X86-G2-SAME: "-target-feature" "+cx16"
! CHECK-X86-G2-SAME: "-target-feature" "+enqcmd"
! CHECK-X86-G2-SAME: "-target-feature" "+f16c"
! CHECK-X86-G2-SAME: "-target-feature" "+fma"
! CHECK-X86-G2-SAME: "-target-feature" "+fma4"
! CHECK-X86-G2-SAME: "-target-feature" "+fsgsbase"
! CHECK-X86-G2-SAME: "-target-feature" "+fxsr"
! CHECK-X86-G2-SAME: "-target-feature" "+invpcid"
!
! RUN: %flang --target=x86_64-linux-gnu -mgfni -mhreset -mkl -mwidekl -mlwp -mlzcnt -mmovbe -mmovdiri -mmovdir64b -mmovrs -mmwaitx -mpku -mpclmul -mpconfig -mpopcnt -mprefetchi -mprfchw -mptwrite -mraoint -mrdpid -mrdpru -mrdrnd -mrtm -mrdseed -msahf -mserialize -msgx -msha -msha512 -msm3 -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-G3
! CHECK-X86-G3: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-G3-SAME: "-target-feature" "+gfni"
! CHECK-X86-G3-SAME: "-target-feature" "+hreset"
! CHECK-X86-G3-SAME: "-target-feature" "+kl"
! CHECK-X86-G3-SAME: "-target-feature" "+widekl"
! CHECK-X86-G3-SAME: "-target-feature" "+lwp"
! CHECK-X86-G3-SAME: "-target-feature" "+lzcnt"
! CHECK-X86-G3-SAME: "-target-feature" "+movbe"
! CHECK-X86-G3-SAME: "-target-feature" "+movdiri"
! CHECK-X86-G3-SAME: "-target-feature" "+movdir64b"
! CHECK-X86-G3-SAME: "-target-feature" "+movrs"
! CHECK-X86-G3-SAME: "-target-feature" "+mwaitx"
! CHECK-X86-G3-SAME: "-target-feature" "+pku"
! CHECK-X86-G3-SAME: "-target-feature" "+pclmul"
! CHECK-X86-G3-SAME: "-target-feature" "+pconfig"
! CHECK-X86-G3-SAME: "-target-feature" "+popcnt"
! CHECK-X86-G3-SAME: "-target-feature" "+prefetchi"
! CHECK-X86-G3-SAME: "-target-feature" "+prfchw"
! CHECK-X86-G3-SAME: "-target-feature" "+ptwrite"
! CHECK-X86-G3-SAME: "-target-feature" "+raoint"
! CHECK-X86-G3-SAME: "-target-feature" "+rdpid"
! CHECK-X86-G3-SAME: "-target-feature" "+rdpru"
! CHECK-X86-G3-SAME: "-target-feature" "+rdrnd"
! CHECK-X86-G3-SAME: "-target-feature" "+rtm"
! CHECK-X86-G3-SAME: "-target-feature" "+rdseed"
! CHECK-X86-G3-SAME: "-target-feature" "+sahf"
! CHECK-X86-G3-SAME: "-target-feature" "+serialize"
! CHECK-X86-G3-SAME: "-target-feature" "+sgx"
! CHECK-X86-G3-SAME: "-target-feature" "+sha"
! CHECK-X86-G3-SAME: "-target-feature" "+sha512"
! CHECK-X86-G3-SAME: "-target-feature" "+sm3"
!
! RUN: %flang --target=x86_64-linux-gnu -msm4 -mtbm -mtsxldtrk -muintr -musermsr -mvaes -mvpclmulqdq -mwaitpkg -mxop -mxsave -mxsavec -mxsaveopt -mxsaves -mshstk -mretpoline-external-thunk -mvzeroupper -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-G4
! CHECK-X86-G4: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-G4-SAME: "-target-feature" "+sm4"
! CHECK-X86-G4-SAME: "-target-feature" "+tbm"
! CHECK-X86-G4-SAME: "-target-feature" "+tsxldtrk"
! CHECK-X86-G4-SAME: "-target-feature" "+uintr"
! CHECK-X86-G4-SAME: "-target-feature" "+usermsr"
! CHECK-X86-G4-SAME: "-target-feature" "+vaes"
! CHECK-X86-G4-SAME: "-target-feature" "+vpclmulqdq"
! CHECK-X86-G4-SAME: "-target-feature" "+waitpkg"
! CHECK-X86-G4-SAME: "-target-feature" "+xop"
! CHECK-X86-G4-SAME: "-target-feature" "+xsave"
! CHECK-X86-G4-SAME: "-target-feature" "+xsavec"
! CHECK-X86-G4-SAME: "-target-feature" "+xsaveopt"
! CHECK-X86-G4-SAME: "-target-feature" "+xsaves"
! CHECK-X86-G4-SAME: "-target-feature" "+shstk"
! CHECK-X86-G4-SAME: "-target-feature" "+retpoline-external-thunk"
! CHECK-X86-G4-SAME: "-target-feature" "+vzeroupper"
!
! RUN: %flang --target=x86_64-linux-gnu -mno-avx -mno-avx2 -mno-sse -mno-aes -mno-amx-tile -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-NEG
! CHECK-X86-NEG: "-fc1" "-triple" "x86_64-unknown-linux-gnu"
! CHECK-X86-NEG-SAME: "-target-feature" "-avx"
! CHECK-X86-NEG-SAME: "-target-feature" "-avx2"
! CHECK-X86-NEG-SAME: "-target-feature" "-sse"
! CHECK-X86-NEG-SAME: "-target-feature" "-aes"
! CHECK-X86-NEG-SAME: "-target-feature" "-amx-tile"
!
! RUN: %flang --target=x86_64-linux-gnu -mapxf -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-APXF
! CHECK-APXF: "-target-feature" "+egpr"
! CHECK-APXF-SAME: "-target-feature" "+ccmp"
!
! RUN: %flang --target=x86_64-linux-gnu -mapx-features=egpr,ndd -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-APX-FEAT
! CHECK-APX-FEAT: "-target-feature" "+egpr"
! CHECK-APX-FEAT-SAME: "-target-feature" "+ndd"
!
! RUN: %flang --target=x86_64-linux-gnu -mno-apx-features=ccmp -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-NO-APX
! CHECK-NO-APX: "-target-feature" "-ccmp"
!
! RUN: %flang --target=x86_64-linux-gnu -mno-gather -mno-scatter -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-X86-GATHER-SCATTER
! CHECK-X86-GATHER-SCATTER: "-target-feature" "+prefer-no-gather"
! CHECK-X86-GATHER-SCATTER-SAME: "-target-feature" "+prefer-no-scatter"
!
! RUN: not %flang --target=aarch64-linux-gnu -mavx -mcrc32 -c %s -### 2>&1 \
! RUN:   | FileCheck %s -check-prefix=CHECK-NONX86
! CHECK-NONX86: error: unsupported option '-mavx' for target 'aarch64-linux-gnu'
! CHECK-NONX86: error: unsupported option '-mcrc32' for target 'aarch64-linux-gnu'
