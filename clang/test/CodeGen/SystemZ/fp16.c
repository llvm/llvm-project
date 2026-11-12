// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -o - %s \
// RUN: | FileCheck %s

void f(__fp16 *a, __fp16 *b, __fp16 *c, __fp16 *d, __fp16 *e) {
  *e = (*a) * (*b) + (*c) * (*d);
}

// CHECK-LABEL: define dso_local void @f(ptr noundef %a, ptr noundef %b, ptr noundef %c, ptr noundef %d, ptr noundef %e) #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %a.addr = alloca ptr, align 8
// CHECK-NEXT:    %b.addr = alloca ptr, align 8
// CHECK-NEXT:    %c.addr = alloca ptr, align 8
// CHECK-NEXT:    %d.addr = alloca ptr, align 8
// CHECK-NEXT:    %e.addr = alloca ptr, align 8
// CHECK-NEXT:    store ptr %a, ptr %a.addr, align 8
// CHECK-NEXT:    store ptr %b, ptr %b.addr, align 8
// CHECK-NEXT:    store ptr %c, ptr %c.addr, align 8
// CHECK-NEXT:    store ptr %d, ptr %d.addr, align 8
// CHECK-NEXT:    store ptr %e, ptr %e.addr, align 8
// CHECK-NEXT:    %0 = load ptr, ptr %a.addr, align 8
// CHECK-NEXT:    %1 = load half, ptr %0, align 2
// CHECK-NEXT:    %conv = fpext half %1 to float
// CHECK-NEXT:    %2 = load ptr, ptr %b.addr, align 8
// CHECK-NEXT:    %3 = load half, ptr %2, align 2
// CHECK-NEXT:    %conv1 = fpext half %3 to float
// CHECK-NEXT:    %mul = fmul float %conv, %conv1
// CHECK-NEXT:    %4 = load ptr, ptr %c.addr, align 8
// CHECK-NEXT:    %5 = load half, ptr %4, align 2
// CHECK-NEXT:    %conv2 = fpext half %5 to float
// CHECK-NEXT:    %6 = load ptr, ptr %d.addr, align 8
// CHECK-NEXT:    %7 = load half, ptr %6, align 2
// CHECK-NEXT:    %conv3 = fpext half %7 to float
// CHECK-NEXT:    %mul4 = fmul float %conv2, %conv3
// CHECK-NEXT:    %add = fadd float %mul, %mul4
// CHECK-NEXT:    %conv5 = fptrunc float %add to half
// CHECK-NEXT:    %8 = load ptr, ptr %e.addr, align 8
// CHECK-NEXT:    store half %conv5, ptr %8, align 2
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
