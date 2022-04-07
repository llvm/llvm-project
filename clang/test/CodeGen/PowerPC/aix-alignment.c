// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc-unknown-aix -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefixes=AIX,AIX32
// RUN: %clang_cc1 -no-opaque-pointers -triple powerpc64-unknown-aix -emit-llvm -o - %s | \
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
// AIX: store double %x, double* %x.addr, align 8
// AIX: load double, double* %x.addr, align 8
// AIX: ret double %0
double retDouble(double x) { return x; }

// AIX32: define void @bar(%struct.StructDouble* noalias sret(%struct.StructDouble) align 4 %agg.result, %struct.StructDouble* noundef byval(%struct.StructDouble) align 4 %x)
// AIX64: define void @bar(%struct.StructDouble* noalias sret(%struct.StructDouble) align 4 %agg.result, %struct.StructDouble* noundef byval(%struct.StructDouble) align 8 %x)
// AIX:     %0 = bitcast %struct.StructDouble* %agg.result to i8*
// AIX:     %1 = bitcast %struct.StructDouble* %x to i8*
// AIX32:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %0, i8* align 4 %1, i32 16, i1 false)
// AIX64:   call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 8 %1, i64 16, i1 false)
StructDouble bar(StructDouble x) { return x; }

// AIX:   define void @foo(double* noundef %out, double* noundef %in)
// AIX32:   %0 = load double*, double** %in.addr, align 4
// AIX64:   %0 = load double*, double** %in.addr, align 8
// AIX:     %1 = load double, double* %0, align 4
// AIX:     %mul = fmul double %1, 2.000000e+00
// AIX32:   %2 = load double*, double** %out.addr, align 4
// AIX64:   %2 = load double*, double** %out.addr, align 8
// AIX:     store double %mul, double* %2, align 4
void foo(double *out, double *in) { *out = *in * 2; }
