// REQUIRES: comgr-has-spirv-translator && comgr-has-spirv-backend

// COM: Compile to SPIR-V via the LLVM SPIR-V backend (handles the
// COM: __builtin_amdgcn_processor_is predicate as a spec constant)
// RUN: %clang --offload-device-only --offload-arch=amdgcnspirv \
// RUN:   --no-gpu-bundle-output -nogpuinc -use-spirv-backend \
// RUN:   -c %s -o %t.spv

// COM: Translate with --isa gfx900: predicate resolves to true,
// COM: gfx900 path is kept, fallback is folded away
// RUN: spirv-translator --isa amdgcn-amd-amdhsa--gfx900 %t.spv -o %t.gfx900.bc
// RUN: %llvm-dis %t.gfx900.bc -o - | %FileCheck --check-prefix=GFX900 %s

// COM: Translate with --isa gfx1010: predicate resolves to false,
// COM: gfx900 path is folded away, fallback is kept
// RUN: spirv-translator --isa amdgcn-amd-amdhsa--gfx1010 %t.spv -o %t.gfx1010.bc
// RUN: %llvm-dis %t.gfx1010.bc -o - | %FileCheck --check-prefix=GFX1010 %s

// GFX900: define {{.*}}void @foo
// GFX900: call {{.*}}void @llvm.trap
// GFX900-NOT: ret void{{$}}

// GFX1010: define {{.*}}void @foo
// GFX1010-NOT: call {{.*}}void @llvm.trap
// GFX1010: ret void

extern "C" __attribute__((device))
void foo() {
    if (__builtin_amdgcn_processor_is("gfx900"))
        __builtin_trap();
}
