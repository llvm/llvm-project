// COM: create a bitcode package
//
// RUN: %clang -c -x hip --offload-arch=gfx900 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx900.bc
//
// RUN: %clang -c -x hip --offload-arch=gfx1030 \
// RUN: -nogpulib -nogpuinc -emit-llvm \
// RUN: --offload-device-only %s -o %t.gfx1030.bc
//
// RUN: llvm-offload-binary \
// RUN: --image=file=%t.gfx900.bc,triple=amdgcn-amd-amdhsa,arch=gfx900,kind=hip \
// RUN: --image=file=%t.gfx1030.bc,triple=amdgcn-amd-amdhsa,arch=gfx1030,kind=hip \
// RUN: -o %t.package.bc
//
// COM: extract using Comgr
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx900 %t.gfx900.output.bc
// RUN: llvm-dis %t.gfx900.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx1030 %t.gfx1030.output.bc
// RUN: llvm-dis %t.gfx1030.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX10 %s
//
// COM: extract a compatible (non-exact) target. the package holds a plain
// COM: gfx900 image, so requesting gfx900:xnack+ misses the exact lookup but
// COM: is still compatible.
// RUN: unpackage %t.package.bc amdgcn-amd-amdhsa gfx900:xnack+ %t.compat.output.bc
// RUN: llvm-dis %t.compat.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// COM: extract from a package whose input data has an empty name. with no name
// COM: provided, unpackage() falls back to a generated unique name.
// RUN: unpackage --name= %t.package.bc amdgcn-amd-amdhsa gfx900 %t.noname.output.bc
// RUN: llvm-dis %t.noname.output.bc -o - | FileCheck --check-prefixes=BOTH,GFX9 %s
//
// COM: failure case 1: input data has a kind other than PACKAGE.
// RUN: unpackage --data-kind=bc --expect-fail \
// RUN: %t.package.bc amdgcn-amd-amdhsa gfx900 %t.wrongkind.output.bc
//
// COM: failure case 2: the package is a PACKAGE but its contents are not a
// COM: valid offload binary.
// COM: (unrecognized magic would instead parse to zero binaries and succeed.)
// RUN: printf 'BC\xc0\xde\xff\xff\xff\xff not a real module' > %t.corrupt.bc
// RUN: unpackage --expect-fail \
// RUN: %t.corrupt.bc amdgcn-amd-amdhsa gfx900 %t.corrupt.output.bc
//
// BOTH: target triple = "amdgcn-amd-amdhsa"
// GFX9: "target-cpu"="gfx900"
// GFX10: "target-cpu"="gfx1030"

__attribute__((device))
void add_value(float* a, float* b, float* res) {
    *res = *a + *b;
}
