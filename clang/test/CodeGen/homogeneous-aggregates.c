// REQUIRES: arm-registered-target,aarch64-registered-target,powerpc-registered-target
// RUN: %clang_cc1 -triple thumbv7-none-none -mfloat-abi hard -x c -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple thumbv7-none-none -mfloat-abi hard -x c++ -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple thumbv7-none-none -mfloat-abi hard -x c++ -DEXTERN_C -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple aarch64-none-none -mfloat-abi hard -x c -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple aarch64-none-none -mfloat-abi hard -x c++ -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple aarch64-none-none -mfloat-abi hard -x c++ -DEXTERN_C -emit-llvm -o - %s | FileCheck %s --check-prefix=AAPCS
// RUN: %clang_cc1 -triple powerpc64le-none-none -mfloat-abi hard -x c -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC --check-prefix=PPC-C
// RUN: %clang_cc1 -triple powerpc64le-none-none -mfloat-abi hard -x c++ -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC --check-prefix=PPC-CXX
// RUN: %clang_cc1 -triple powerpc64le-none-none -mfloat-abi hard -x c++ -DEXTERN_C -emit-llvm -o - %s | FileCheck %s --check-prefix=PPC --check-prefix=PPC-CXX

// The aim here is to test whether each of these structure types is
// regarded as a homogeneous aggregate of a single kind of
// floating-point item, because in all of these ABIs, that changes the
// calling convention.
//
// We expect that 'Floats' and 'Doubles' are homogeneous, and 'Mixed'
// is not. But the next two structures, with separating zero-size
// bitfields, are more interesting.
//
// For the Arm architecture, AAPCS says that the homogeneity rule is
// applied _after_ data layout is completed, so that it's unaffected
// by anything that was completely discarded during data layout. So we
// expect that FloatsBF and DoublesBF still count as homogeneous.
//
// But on PowerPC, it depends on whether the source language is C or
// C++, because that's consistent with the decisions gcc makes.

struct Floats {
    float a;
    float b;
};

struct Doubles {
    double a;
    double b;
};

struct Mixed {
    double a;
    float b;
};

struct FloatsBF {
    float a;
    int : 0;
    float b;
};

struct DoublesBF {
    double a;
    int : 0;
    double b;
};

// In C++ mode, we test both with and without extern "C", to ensure
// that doesn't make a difference.
#ifdef EXTERN_C
#define LINKAGE extern "C"
#else
#define LINKAGE
#endif

// For Arm backends, the IR emitted for the homogeneous-aggregate
// return convention uses the actual structure type, so that
// HandleFloats returns a %struct.Floats, and so on. To check that
// 'Mixed' is not treated as homogeneous, it's enough to check that
// its return type is _not_ %struct.Mixed. (The fallback handling
// varies between AArch32 and AArch64.)
//
// For PowerPC, homogeneous structure types are lowered to an IR array
// types like [2 x float], and the non-homogeneous Mixed is lowered to
// a pair of i64.

// AAPCS: define{{.*}} %struct.Floats @{{.*HandleFloats.*}}
// PPC: define{{.*}} [2 x float] @{{.*HandleFloats.*}}
LINKAGE struct Floats HandleFloats(struct Floats x) { return x; }

// AAPCS: define{{.*}} %struct.Doubles @{{.*HandleDoubles.*}}
// PPC: define{{.*}} [2 x double] @{{.*HandleDoubles.*}}
LINKAGE struct Doubles HandleDoubles(struct Doubles x) { return x; }

// AAPCS-NOT: define{{.*}} %struct.Mixed @{{.*HandleMixed.*}}
// PPC: define{{.*}} { i64, i64 } @{{.*HandleMixed.*}}
LINKAGE struct Mixed HandleMixed(struct Mixed x) { return x; }

// AAPCS: define{{.*}} %struct.FloatsBF @{{.*HandleFloatsBF.*}}
// PPC-C-NOT: define{{.*}} [2 x float] @{{.*HandleFloatsBF.*}}
// PPC-CXX: define{{.*}} [2 x float] @{{.*HandleFloatsBF.*}}
LINKAGE struct FloatsBF HandleFloatsBF(struct FloatsBF x) { return x; }

// AAPCS: define{{.*}} %struct.DoublesBF @{{.*HandleDoublesBF.*}}
// PPC-C-NOT: define{{.*}} [2 x double] @{{.*HandleDoublesBF.*}}
// PPC-CXX: define{{.*}} [2 x double] @{{.*HandleDoublesBF.*}}
LINKAGE struct DoublesBF HandleDoublesBF(struct DoublesBF x) { return x; }
