// RUN: %clang_cc1 -triple aarch64 -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple aarch64 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple aarch64 -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s

__attribute__((target("arch=armv8.2-a")))
void v82(void) {}

// CIR:      cir.func{{.*}} @v82()
// CIR-SAME: "cir.target-features" = "+crc,+fp-armv8,+lse,+neon,+ras,+rdm,+v8.1a,+v8.2a,+v8a"
// CIR-NOT:  "cir.target-cpu"
// LLVM-DAG: define{{.*}} void @v82(){{.*}} #[[ATTR_V82:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V82]] = {{.*}}"target-features"="+crc,+fp-armv8,+lse,+neon,+ras,+rdm,+v8.1a,+v8.2a,+v8a"

// target("arch=armv8.2-a+sve"): arch with SVE extension.
__attribute__((target("arch=armv8.2-a+sve")))
void v82sve(void) {}

// CIR:      cir.func{{.*}} @v82sve()
// CIR-SAME: "cir.target-features" = "+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rdm,+sve,+v8.1a,+v8.2a,+v8a"
// LLVM-DAG: define{{.*}} void @v82sve(){{.*}} #[[ATTR_V82SVE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V82SVE]] = {{.*}}"target-features"="+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rdm,+sve,+v8.1a,+v8.2a,+v8a"

// target("arch=armv8.2-a+sve2"): arch + sve2 implies +sve.
__attribute__((target("arch=armv8.2-a+sve2")))
void v82sve2(void) {}

// CIR:      cir.func{{.*}} @v82sve2()
// CIR-SAME: "cir.target-features" = "+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rdm,+sve,+sve2,+v8.1a,+v8.2a,+v8a"
// LLVM-DAG: define{{.*}} void @v82sve2(){{.*}} #[[ATTR_V82SVE2:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V82SVE2]] = {{.*}}"target-features"="+crc,+fp-armv8,+fullfp16,+lse,+neon,+ras,+rdm,+sve,+sve2,+v8.1a,+v8.2a,+v8a"

// target("arch=armv8.2-a+sve+sve2"): same effective feature set as v82sve2;
// reuses ATTR_V82SVE2.
__attribute__((target("arch=armv8.2-a+sve+sve2")))
void v82svesve2(void) {}

// LLVM-DAG: define{{.*}} void @v82svesve2(){{.*}} #[[ATTR_V82SVE2]]

// target("arch=armv8.6-a+sve2"): later baseline + sve2.
__attribute__((target("arch=armv8.6-a+sve2")))
void v86sve2(void) {}

// CIR:      cir.func{{.*}} @v86sve2()
// CIR-SAME: "cir.target-features" = "+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+predres,+ras,+rcpc,+rdm,+sb,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a"
// LLVM-DAG: define{{.*}} void @v86sve2(){{.*}} #[[ATTR_V86SVE2:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V86SVE2]] = {{.*}}"target-features"="+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+predres,+ras,+rcpc,+rdm,+sb,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a"

// target("cpu=cortex-a710"): cpu override pulls in cortex-a710's baseline.
__attribute__((target("cpu=cortex-a710")))
void a710(void) {}

// CIR:      cir.func{{.*}} @a710()
// CIR-SAME: "cir.target-cpu" = "cortex-a710"
// CIR-SAME: "cir.target-features" = "+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+ete,+flagm,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+mte,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+ssbs,+sve,+sve-bitperm,+sve2,+trbe,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+v9a"
// LLVM-DAG: define{{.*}} void @a710(){{.*}} #[[ATTR_A710:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_A710]] = {{.*}}"target-cpu"="cortex-a710"{{.*}}"target-features"="+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+ete,+flagm,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+mte,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+ssbs,+sve,+sve-bitperm,+sve2,+trbe,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+v9a"

// target("tune=cortex-a710"): only tune-cpu changes; no target-cpu, no
// target-features (global features unchanged from base aarch64 defaults).
__attribute__((target("tune=cortex-a710")))
void tunea710(void) {}

// CIR:      cir.func{{.*}} @tunea710()
// CIR-SAME: "cir.tune-cpu" = "cortex-a710"
// CIR-NOT:  "cir.target-cpu"
// CIR-NOT:  "cir.target-features"
// LLVM-DAG: define{{.*}} void @tunea710(){{.*}} #[[ATTR_TUNEA710:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_TUNEA710]] = {{.*}}"tune-cpu"="cortex-a710"

// target("cpu=generic"): generic cpu pulls in its small feature set.
__attribute__((target("cpu=generic")))
void generic(void) {}

// CIR:      cir.func{{.*}} @generic()
// CIR-SAME: "cir.target-cpu" = "generic"
// CIR-SAME: "cir.target-features" = "+ete,+fp-armv8,+neon,+trbe,+v8a"
// LLVM-DAG: define{{.*}} void @generic(){{.*}} #[[ATTR_GENERIC:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_GENERIC]] = {{.*}}"target-cpu"="generic"{{.*}}"target-features"="+ete,+fp-armv8,+neon,+trbe,+v8a"

// target("tune=generic"): only tune-cpu set.
__attribute__((target("tune=generic")))
void tune(void) {}

// CIR:      cir.func{{.*}} @tune()
// CIR-SAME: "cir.tune-cpu" = "generic"
// CIR-NOT:  "cir.target-cpu"
// CIR-NOT:  "cir.target-features"
// LLVM-DAG: define{{.*}} void @tune(){{.*}} #[[ATTR_TUNE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_TUNE]] = {{.*}}"tune-cpu"="generic"

// target("cpu=neoverse-n1,tune=cortex-a710"): both cpu and tune set.
__attribute__((target("cpu=neoverse-n1,tune=cortex-a710")))
void n1tunea710(void) {}

// CIR:      cir.func{{.*}} @n1tunea710()
// CIR-SAME: "cir.target-cpu" = "neoverse-n1"
// CIR-SAME: "cir.target-features" = "+aes,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+perfmon,+ras,+rcpc,+rdm,+sha2,+spe,+ssbs,+v8.1a,+v8.2a,+v8a"
// CIR-SAME: "cir.tune-cpu" = "cortex-a710"
// LLVM-DAG: define{{.*}} void @n1tunea710(){{.*}} #[[ATTR_N1TUNE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_N1TUNE]] = {{.*}}"target-cpu"="neoverse-n1"{{.*}}"target-features"="+aes,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+perfmon,+ras,+rcpc,+rdm,+sha2,+spe,+ssbs,+v8.1a,+v8.2a,+v8a"{{.*}}"tune-cpu"="cortex-a710"

// target("sve,tune=cortex-a710"): feature add + tune. No cpu override.
__attribute__((target("sve,tune=cortex-a710")))
void svetunea710(void) {}

// CIR:      cir.func{{.*}} @svetunea710()
// CIR-SAME: "cir.target-features" = "+fp-armv8,+fullfp16,+sve"
// CIR-SAME: "cir.tune-cpu" = "cortex-a710"
// CIR-NOT:  "cir.target-cpu"
// LLVM-DAG: define{{.*}} void @svetunea710(){{.*}} #[[ATTR_SVETUNE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_SVETUNE]] = {{.*}}"target-features"="+fp-armv8,+fullfp16,+sve"{{.*}}"tune-cpu"="cortex-a710"

// target("+sve,tune=cortex-a710"): explicit "+" prefix; same effect; reuses
// ATTR_SVETUNE.
__attribute__((target("+sve,tune=cortex-a710")))
void plussvetunea710(void) {}

// LLVM-DAG: define{{.*}} void @plussvetunea710(){{.*}} #[[ATTR_SVETUNE]]

// target("cpu=neoverse-v1,+sve2"): cpu + extra feature.
__attribute__((target("cpu=neoverse-v1,+sve2")))
void v1plussve2(void) {}

// CIR:      cir.func{{.*}} @v1plussve2()
// CIR-SAME: "cir.target-cpu" = "neoverse-v1"
// CIR-SAME: "cir.target-features" = "+aes,+bf16,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a"
// LLVM-DAG: define{{.*}} void @v1plussve2(){{.*}} #[[ATTR_V1SVE2:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V1SVE2]] = {{.*}}"target-cpu"="neoverse-v1"{{.*}}"target-features"="+aes,+bf16,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a"

// target("cpu=neoverse-v1+sve2"): cpu+feature without comma; same effect;
// reuses ATTR_V1SVE2.
__attribute__((target("cpu=neoverse-v1+sve2")))
void v1sve2(void) {}

// LLVM-DAG: define{{.*}} void @v1sve2(){{.*}} #[[ATTR_V1SVE2]]

// target("cpu=neoverse-v1,+nosve"): cpu + feature negation via "+no" prefix.
__attribute__((target("cpu=neoverse-v1,+nosve")))
void v1minussve(void) {}

// CIR:      cir.func{{.*}} @v1minussve()
// CIR-SAME: "cir.target-cpu" = "neoverse-v1"
// CIR-SAME: "cir.target-features" = "+aes,+bf16,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,-sve"
// LLVM-DAG: define{{.*}} void @v1minussve(){{.*}} #[[ATTR_V1NOSVE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_V1NOSVE]] = {{.*}}"target-cpu"="neoverse-v1"{{.*}}"target-features"="+aes,+bf16,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+rand,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+spe,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,-sve"

// target("cpu=neoverse-v1,no-sve"): "no-" prefix; same effect; reuses
// ATTR_V1NOSVE.
__attribute__((target("cpu=neoverse-v1,no-sve")))
void v1nosve(void) {}

// LLVM-DAG: define{{.*}} void @v1nosve(){{.*}} #[[ATTR_V1NOSVE]]

// target("cpu=neoverse-v1+nosve"): cpu+nosve without comma; same effect;
// reuses ATTR_V1NOSVE.
__attribute__((target("cpu=neoverse-v1+nosve")))
void v1msve(void) {}

// LLVM-DAG: define{{.*}} void @v1msve(){{.*}} #[[ATTR_V1NOSVE]]

// target("+sve"): single feature add; no cpu/tune override.
__attribute__((target("+sve")))
void plussve(void) {}

// CIR:      cir.func{{.*}} @plussve()
// CIR-SAME: "cir.target-features" = "+fp-armv8,+fullfp16,+sve"
// CIR-NOT:  "cir.target-cpu"
// LLVM-DAG: define{{.*}} void @plussve(){{.*}} #[[ATTR_PLUSSVE:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_PLUSSVE]] = {{.*}}"target-features"="+fp-armv8,+fullfp16,+sve"

// target("+sve+nosve2"): chained features; nosve2 is a no-op; reuses
// ATTR_PLUSSVE.
__attribute__((target("+sve+nosve2")))
void plussveplussve2(void) {}

// LLVM-DAG: define{{.*}} void @plussveplussve2(){{.*}} #[[ATTR_PLUSSVE]]

// target("sve,no-sve2"): comma-separated equivalent; same effect; reuses
// ATTR_PLUSSVE.
__attribute__((target("sve,no-sve2")))
void plussveminusnosve2(void) {}

// LLVM-DAG: define{{.*}} void @plussveminusnosve2(){{.*}} #[[ATTR_PLUSSVE]]

// target("+fp16"): just adds fp16.
__attribute__((target("+fp16")))
void plusfp16(void) {}

// CIR:      cir.func{{.*}} @plusfp16()
// CIR-SAME: "cir.target-features" = "+fp-armv8,+fullfp16"
// CIR-NOT:  "cir.target-cpu"
// LLVM-DAG: define{{.*}} void @plusfp16(){{.*}} #[[ATTR_FP16:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_FP16]] = {{.*}}"target-features"="+fp-armv8,+fullfp16"

// target("cpu=neoverse-n1,tune=cortex-a710,arch=armv8.6-a+sve2"): everything
// at once. arch overrides do NOT clear cpu= here.
__attribute__((target("cpu=neoverse-n1,tune=cortex-a710,arch=armv8.6-a+sve2")))
void all(void) {}

// CIR:      cir.func{{.*}} @all()
// CIR-SAME: "cir.target-cpu" = "neoverse-n1"
// CIR-SAME: "cir.target-features" = "+aes,+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+spe,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a"
// CIR-SAME: "cir.tune-cpu" = "cortex-a710"
// LLVM-DAG: define{{.*}} void @all(){{.*}} #[[ATTR_ALL:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_ALL]] = {{.*}}"target-cpu"="neoverse-n1"{{.*}}"target-features"="+aes,+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+spe,+ssbs,+sve,+sve2,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8a"{{.*}}"tune-cpu"="cortex-a710"

// target("+nosimd"): "+no" prefix for simd; produces no target-features (the
// negation cancels the default neon, leaving an empty effective delta).
__attribute__((target("+nosimd")))
void plusnosimd(void) {}

// CIR:      cir.func{{.*}} @plusnosimd()
// CIR-NOT:  "cir.target-cpu"
// CIR-NOT:  "cir.target-features"
// LLVM-DAG: define{{.*}} void @plusnosimd(){{.*}} #[[ATTR_NOSIMD:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_NOSIMD]] = { {{.*}} }

// target("no-simd"): equivalent "no-" syntax; reuses ATTR_NOSIMD.
__attribute__((target("no-simd")))
void nosimd(void) {}

// LLVM-DAG: define{{.*}} void @nosimd(){{.*}} #[[ATTR_NOSIMD]]

// target("no-v9.3a"): disable an arch-level feature without enabling anything.
__attribute__((target("no-v9.3a")))
void minusarch(void) {}

// CIR:      cir.func{{.*}} @minusarch()
// CIR-SAME: "cir.target-features" = "-v9.3a"
// LLVM-DAG: define{{.*}} void @minusarch(){{.*}} #[[ATTR_MINUSARCH:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_MINUSARCH]] = {{.*}}"target-features"="-v9.3a"

// target("cpu=apple-m4"): another cpu with a large feature set.
__attribute__((target("cpu=apple-m4")))
void applem4(void) {}

// CIR:      cir.func{{.*}} @applem4()
// CIR-SAME: "cir.target-cpu" = "apple-m4"
// CIR-SAME: "cir.target-features" = "+aes,+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+spe-eef,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8.7a,+v8a,+wfxt"
// LLVM-DAG: define{{.*}} void @applem4(){{.*}} #[[ATTR_APPLEM4:[0-9]+]]
// LLVM-DAG: attributes #[[ATTR_APPLEM4]] = {{.*}}"target-cpu"="apple-m4"{{.*}}"target-features"="+aes,+bf16,+bti,+ccidx,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fpac,+fullfp16,+i8mm,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+sme,+sme-f64f64,+sme-i16i64,+sme2,+spe-eef,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8.6a,+v8.7a,+v8a,+wfxt"
