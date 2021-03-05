# REQUIRES: x86, aarch64

# RUN: mkdir -p %t
# RUN: llvm-mc -filetype obj -triple arm64-apple-ios14.4 %s -o %t/arm64-ios.o
# RUN: not %lld -dylib -arch arm64 -platform_version ios 10 11 -o /dev/null \
# RUN:   -lSystem %S/Inputs/libStubLink.tbd %t/arm64-ios.o 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype obj -triple x86_64-apple-iossimulator14.4 %s -o %t/x86_64-sim.o
# RUN: not %lld -dylib -arch x86_64 -platform_version ios-simulator 10 11 -o /dev/null \
# RUN:   -lSystem %S/Inputs/libStubLink.tbd %t/x86_64-sim.o 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype obj -triple arm64-apple-iossimulator14.4 %s -o %t/arm64-sim.o
# RUN: %lld -dylib -arch arm64 -platform_version ios-simulator 10 11 -o \
# RUN:   /dev/null %S/Inputs/libStubLink.tbd %t/arm64-sim.o

# CHECK: error: undefined symbol: _arm64_sim_only

.data
.quad _arm64_sim_only
