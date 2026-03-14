// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX32
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX64

// AIX: @d = global double 0.000000e+00, align 8
double d;

typedef struct {
  double d;
  int i;
} StructDouble;

// AIX: @d1 = global %struct.StructDouble zeroinitializer, align 8
StructDouble d1;

// AIX: double @retDouble(double noundef %x)
// AIX: %x.addr = alloca double, align 8
// AIX: store double %x, ptr %x.addr, align 8
// AIX: load double, ptr %x.addr, align 8
// AIX: ret double %0
double retDouble(double x) { return x; }

// AIX32: define void @bar(ptr dead_on_unwind noalias writable sret(%struct.StructDouble) align 4 %agg.result, ptr noundef byval(%struct.StructDouble) align 4 %x)
// AIX64: define void @bar(ptr dead_on_unwind noalias writable sret(%struct.StructDouble) align 4 %agg.result, ptr noundef byval(%struct.StructDouble) align 8 %x)
// AIX32:   call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.result, ptr align 4 %x, i32 16, i1 false)
// AIX64:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.result, ptr align 8 %x, i64 16, i1 false)
StructDouble bar(StructDouble x) { return x; }

// AIX:   define void @foo(ptr noundef %out, ptr noundef %in)
// AIX32:   %0 = load ptr, ptr %in.addr, align 4
// AIX64:   %0 = load ptr, ptr %in.addr, align 8
// AIX:     %1 = load double, ptr %0, align 4
// AIX:     %mul = fmul double %1, 2.000000e+00
// AIX32:   %2 = load ptr, ptr %out.addr, align 4
// AIX64:   %2 = load ptr, ptr %out.addr, align 8
// AIX:     store double %mul, ptr %2, align 4
void foo(double *out, double *in) { *out = *in * 2; }
