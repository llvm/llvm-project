! REQUIRES: target=powerpc{{.*}}
! RUN: %flang_fc1 -emit-fir -target-cpu pwr10 %s -o - | FileCheck %s --check-prefixes=ALL,FEATURE
! RUN: %flang_fc1 -emit-fir -target-cpu pwr10 -target-feature +privileged %s -o - | FileCheck %s --check-prefixes=ALL,BOTH

! ALL: module attributes {

! ALL: fir.target_cpu = "pwr10"

! FEATURE: fir.target_features = #llvm.target_features<[
! FEATURE: "+64bit-support", "+allow-unaligned-fp-access", "+altivec", "+bpermd", "+cmpb", "+crbits", "+crypto", "+direct-move", "+extdiv", "+fast-MFLR", "+fcpsgn", "+fpcvt", "+fprnd", "+fpu", "+fre", "+fres", "+frsqrte", "+frsqrtes", "+fsqrt", "+fuse-add-logical", "+fuse-arith-add", "+fuse-logical", "+fuse-logical-add", "+fuse-sha3", "+fuse-store", "+fusion", "+hard-float", "+icbt", "+isa-v206-instructions", "+isa-v207-instructions", "+isa-v30-instructions", "+isa-v31-instructions", "+isel", "+ldbrx", "+lfiwax", "+mfocrf", "+mma", "+paired-vector-memops", "+partword-atomics", "+pcrelative-memops", "+popcntd", "+power10-vector", "+power8-altivec", "+power8-vector", "+power9-altivec", "+power9-vector", "+ppc-postra-sched", "+ppc-prera-sched", "+predictable-select-expensive", "+prefix-instrs", "+quadword-atomics", "+recipprec", "+stfiwx", "+two-const-nr", "+vsx"
! FEATURE: ]>

! BOTH: fir.target_features = #llvm.target_features<[
! BOTH: "+64bit-support", "+allow-unaligned-fp-access", "+altivec", "+bpermd", "+cmpb", "+crbits", "+crypto", "+direct-move", "+extdiv", "+fast-MFLR", "+fcpsgn", "+fpcvt", "+fprnd", "+fpu", "+fre", "+fres", "+frsqrte", "+frsqrtes", "+fsqrt", "+fuse-add-logical", "+fuse-arith-add", "+fuse-logical", "+fuse-logical-add", "+fuse-sha3", "+fuse-store", "+fusion", "+hard-float", "+icbt", "+isa-v206-instructions", "+isa-v207-instructions", "+isa-v30-instructions", "+isa-v31-instructions", "+isel", "+ldbrx", "+lfiwax", "+mfocrf", "+mma", "+paired-vector-memops", "+partword-atomics", "+pcrelative-memops", "+popcntd", "+power10-vector", "+power8-altivec", "+power8-vector", "+power9-altivec", "+power9-vector", "+ppc-postra-sched", "+ppc-prera-sched", "+predictable-select-expensive", "+prefix-instrs", "+privileged", "+quadword-atomics", "+recipprec", "+stfiwx", "+two-const-nr", "+vsx"
! BOTH: ]>
