// RUN: %clang_cc1 -triple s390x-linux-gnu \
// RUN: -ffloat16-excess-precision=standard -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefix=STANDARD

// RUN: %clang_cc1 -triple s390x-linux-gnu \
// RUN: -ffloat16-excess-precision=none -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefix=NONE

// RUN: %clang_cc1 -triple s390x-linux-gnu \
// RUN: -ffloat16-excess-precision=fast -emit-llvm -o - %s \
// RUN: | FileCheck %s -check-prefix=FAST

_Float16 f(_Float16 a, _Float16 b, _Float16 c, _Float16 d) {
    return a * b + c * d;
}

// STANDARD-LABEL: define dso_local half @f(half noundef %a, half noundef %b, half noundef %c, half noundef %d) #0 {
// STANDARD-NEXT:  entry:
// STANDARD-NEXT:    %a.addr = alloca half, align 2
// STANDARD-NEXT:    %b.addr = alloca half, align 2
// STANDARD-NEXT:    %c.addr = alloca half, align 2
// STANDARD-NEXT:    %d.addr = alloca half, align 2
// STANDARD-NEXT:    store half %a, ptr %a.addr, align 2
// STANDARD-NEXT:    store half %b, ptr %b.addr, align 2
// STANDARD-NEXT:    store half %c, ptr %c.addr, align 2
// STANDARD-NEXT:    store half %d, ptr %d.addr, align 2
// STANDARD-NEXT:    %0 = load half, ptr %a.addr, align 2
// STANDARD-NEXT:    %ext = fpext half %0 to float
// STANDARD-NEXT:    %1 = load half, ptr %b.addr, align 2
// STANDARD-NEXT:    %ext1 = fpext half %1 to float
// STANDARD-NEXT:    %mul = fmul float %ext, %ext1
// STANDARD-NEXT:    %2 = load half, ptr %c.addr, align 2
// STANDARD-NEXT:    %ext2 = fpext half %2 to float
// STANDARD-NEXT:    %3 = load half, ptr %d.addr, align 2
// STANDARD-NEXT:    %ext3 = fpext half %3 to float
// STANDARD-NEXT:    %mul4 = fmul float %ext2, %ext3
// STANDARD-NEXT:    %add = fadd float %mul, %mul4
// STANDARD-NEXT:    %unpromotion = fptrunc float %add to half
// STANDARD-NEXT:    ret half %unpromotion
// STANDARD-NEXT:  }

// NONE-LABEL: define dso_local half @f(half noundef %a, half noundef %b, half noundef %c, half noundef %d) #0 {
// NONE-NEXT:  entry:
// NONE-NEXT:    %a.addr = alloca half, align 2
// NONE-NEXT:    %b.addr = alloca half, align 2
// NONE-NEXT:    %c.addr = alloca half, align 2
// NONE-NEXT:    %d.addr = alloca half, align 2
// NONE-NEXT:    store half %a, ptr %a.addr, align 2
// NONE-NEXT:    store half %b, ptr %b.addr, align 2
// NONE-NEXT:    store half %c, ptr %c.addr, align 2
// NONE-NEXT:    store half %d, ptr %d.addr, align 2
// NONE-NEXT:    %0 = load half, ptr %a.addr, align 2
// NONE-NEXT:    %1 = load half, ptr %b.addr, align 2
// NONE-NEXT:    %mul = fmul half %0, %1
// NONE-NEXT:    %2 = load half, ptr %c.addr, align 2
// NONE-NEXT:    %3 = load half, ptr %d.addr, align 2
// NONE-NEXT:    %mul1 = fmul half %2, %3
// NONE-NEXT:    %add = fadd half %mul, %mul1
// NONE-NEXT:    ret half %add
// NONE-NEXT:  }

// FAST-LABEL: define dso_local half @f(half noundef %a, half noundef %b, half noundef %c, half noundef %d) #0 {
// FAST-NEXT:  entry:
// FAST-NEXT:    %a.addr = alloca half, align 2
// FAST-NEXT:    %b.addr = alloca half, align 2
// FAST-NEXT:    %c.addr = alloca half, align 2
// FAST-NEXT:    %d.addr = alloca half, align 2
// FAST-NEXT:    store half %a, ptr %a.addr, align 2
// FAST-NEXT:    store half %b, ptr %b.addr, align 2
// FAST-NEXT:    store half %c, ptr %c.addr, align 2
// FAST-NEXT:    store half %d, ptr %d.addr, align 2
// FAST-NEXT:    %0 = load half, ptr %a.addr, align 2
// FAST-NEXT:    %ext = fpext half %0 to float
// FAST-NEXT:    %1 = load half, ptr %b.addr, align 2
// FAST-NEXT:    %ext1 = fpext half %1 to float
// FAST-NEXT:    %mul = fmul float %ext, %ext1
// FAST-NEXT:    %2 = load half, ptr %c.addr, align 2
// FAST-NEXT:    %ext2 = fpext half %2 to float
// FAST-NEXT:    %3 = load half, ptr %d.addr, align 2
// FAST-NEXT:    %ext3 = fpext half %3 to float
// FAST-NEXT:    %mul4 = fmul float %ext2, %ext3
// FAST-NEXT:    %add = fadd float %mul, %mul4
// FAST-NEXT:    %unpromotion = fptrunc float %add to half
// FAST-NEXT:    ret half %unpromotion
// FAST-NEXT:  }
