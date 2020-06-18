// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=AIX32 %s
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=AIX64 %s

typedef struct {
} Zero;
typedef struct {
  char c;
} One;
typedef struct {
  short s;
} Two;
typedef struct {
  char c[3];
} Three;
typedef struct {
  float f;
} Four;
typedef struct {
  char c[5];
} Five;
typedef struct {
  short s[3];
} Six;
typedef struct {
  char c[7];
} Seven;
typedef struct {
  long long l;
} Eight;
typedef struct {
  int i;
} __attribute__((aligned(32))) OverAligned;
typedef struct {
  int i;
  vector signed int vsi;
} StructVector;

// AIX32-LABEL: define void @arg0(%struct.Zero* byval(%struct.Zero) align 4 %x)
// AIX64-LABEL: define void @arg0(%struct.Zero* byval(%struct.Zero) align 8 %x)
void arg0(Zero x) {}

// AIX32-LABEL: define void @arg1(%struct.One* byval(%struct.One) align 4 %x)
// AIX64-LABEL: define void @arg1(%struct.One* byval(%struct.One) align 8 %x)
void arg1(One x) {}

// AIX32-LABEL: define void @arg2(%struct.Two* byval(%struct.Two) align 4 %x)
// AIX64-LABEL: define void @arg2(%struct.Two* byval(%struct.Two) align 8 %x)
void arg2(Two x) {}

// AIX32-LABEL: define void @arg3(%struct.Three* byval(%struct.Three) align 4 %x)
// AIX64-LABEL: define void @arg3(%struct.Three* byval(%struct.Three) align 8 %x)
void arg3(Three x) {}

// AIX32-LABEL: define void @arg4(%struct.Four* byval(%struct.Four) align 4 %x)
// AIX64-LABEL: define void @arg4(%struct.Four* byval(%struct.Four) align 8 %x)
void arg4(Four x) {}

// AIX32-LABEL: define void @arg5(%struct.Five* byval(%struct.Five) align 4 %x)
// AIX64-LABEL: define void @arg5(%struct.Five* byval(%struct.Five) align 8 %x)
void arg5(Five x) {}

// AIX32-LABEL: define void @arg6(%struct.Six* byval(%struct.Six) align 4 %x)
// AIX64-LABEL: define void @arg6(%struct.Six* byval(%struct.Six) align 8 %x)
void arg6(Six x) {}

// AIX32-LABEL: define void @arg7(%struct.Seven* byval(%struct.Seven) align 4 %x)
// AIX64-LABEL: define void @arg7(%struct.Seven* byval(%struct.Seven) align 8 %x)
void arg7(Seven x) {}

// AIX32-LABEL: define void @arg8(%struct.Eight* byval(%struct.Eight) align 4 %0)
// AIX32:         %x = alloca %struct.Eight, align 8
// AIX32:         call void @llvm.memcpy.p0i8.p0i8.i32
// AIX64-LABEL: define void @arg8(%struct.Eight* byval(%struct.Eight) align 8 %x)
void arg8(Eight x) {}

// AIX32-LABEL: define void @arg9(%struct.OverAligned* byval(%struct.OverAligned) align 4 %0)
// AIX32:         %x = alloca %struct.OverAligned, align 32
// AIX32:         call void @llvm.memcpy.p0i8.p0i8.i32
// AIX64-LABEL: define void @arg9(%struct.OverAligned* byval(%struct.OverAligned) align 8 %0)
// AIX64:         %x = alloca %struct.OverAligned, align 32
// AIX64:         call void @llvm.memcpy.p0i8.p0i8.i64
void arg9(OverAligned x) {}

// AIX32-LABEL: define void @arg10(%struct.StructVector* byval(%struct.StructVector) align 16 %x)
// AIX64-LABEL: define void @arg10(%struct.StructVector* byval(%struct.StructVector) align 16 %x)
void arg10(StructVector x) {}
