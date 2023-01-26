// RUN: %clang_cc1 -triple x86_64 -emit-llvm -o - %s | FileCheck %s '-D$MANGLE_AS=p0' '-D$CONST_AS=' --check-prefixes=CHECK,X86
// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s | FileCheck %s '-D$MANGLE_AS=p4' '-D$CONST_AS= addrspace(4)' --check-prefixes=CHECK,AMDGPU
// END.

// CHECK: private unnamed_addr[[$CONST_AS]] constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr[[$CONST_AS]] constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"

struct foo {
    int v __attribute__((annotate("v_ann_0"))) __attribute__((annotate("v_ann_1")));
};

static struct foo gf;

int main(int argc, char **argv) {
    struct foo f;
    f.v = argc;
// CHECK: getelementptr inbounds %struct.foo, ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0.[[$MANGLE_AS]]({{.*}}str{{.*}}str{{.*}}i32 9, ptr[[$CONST_AS]] null)
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0.[[$MANGLE_AS]]({{.*}}str{{.*}}str{{.*}}i32 9, ptr[[$CONST_AS]] null)
    gf.v = argc;
// X86: call ptr @llvm.ptr.annotation.p0.p0(ptr @gf, ptr @.str{{.*}}, ptr @.str{{.*}}, i32 9, ptr null)
// X86-NEXT: call ptr @llvm.ptr.annotation.p0.p0(ptr %{{.*}}, ptr @.str{{.*}}, ptr @.str{{.*}}, i32 9, ptr null)
// AMDGPU: call ptr @llvm.ptr.annotation.p0.p4(ptr addrspacecast (ptr addrspace(1) @gf to ptr), ptr addrspace(4) @.str{{.*}}, ptr addrspace(4) @.str{{.*}}, i32 9, ptr addrspace(4) null)
// AMDGPU-NEXT: call ptr @llvm.ptr.annotation.p0.p4(ptr %{{.*}}, ptr addrspace(4) @.str{{.*}}, ptr addrspace(4) @.str{{.*}}, i32 9, ptr addrspace(4) null)
    return 0;
}
