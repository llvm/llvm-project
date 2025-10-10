// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -O1 -fstrict-enums -std=c++11 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple i386-unknown-unknown -emit-llvm -O3 -std=c++11 -o - | FileCheck --check-prefixes=NO-STRICT-ENUMS-C,NO-STRICT-ENUMS %s
// RUN: %clang_cc1 -x c %s -triple i386-unknown-unknown -emit-llvm -O1 -std=c23 -fstrict-enums -o - | FileCheck --check-prefix=NO-STRICT-ENUMS-C %s

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef __cplusplus
enum e1 { };
bool g1a(enum e1 x) {
  return (unsigned int)x > 1;
}
// CHECK-LABEL: define{{.*}} i1 @g1a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-LABEL: define{{.*}} i1 @g1a
// NO-STRICT-ENUMS: ret i1 %cmp
bool g1b(enum e1 x) {
  return (unsigned int)x >= 1;
}
// CHECK-LABEL: define{{.*}} i1 @g1b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-LABEL: define{{.*}} i1 @g1b
// NO-STRICT-ENUMS: ret i1 %cmp
#endif // __cplusplus

enum e2 { e2_a };
bool g2a(enum e2 x) {
  return (unsigned int)x > 1;
}
// CHECK-LABEL: define{{.*}} i1 @g2a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g2a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g2b(enum e2 x) {
  return (unsigned int)x >= 1;
}
// CHECK-LABEL: define{{.*}} i1 @g2b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g2b
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e3 { e3_a = 1 };
bool g3a(enum e3 x) {
  return (unsigned int)x > 1;
}
// CHECK-LABEL: define{{.*}} i1 @g3a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g3a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g3b(enum e3 x) {
  return (unsigned int)x >= 1;
}
// CHECK-LABEL: define{{.*}} i1 @g3b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g3b
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e4 { e4_a = -1 };
bool g4a(enum e4 x) {
  return (int)x > 0;
}
// CHECK-LABEL: define{{.*}} i1 @g4a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g4a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g4b(enum e4 x) {
  return (int)x >= 0;
}
// CHECK-LABEL: define{{.*}} i1 @g4b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g4b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g4c(enum e4 x) {
  return (int)x < -1;
}
// CHECK-LABEL: define{{.*}} i1 @g4c
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g4c
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g4d(enum e4 x) {
  return (int)x <= -1;
}
// CHECK-LABEL: define{{.*}} i1 @g4d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g4d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e5 { e5_a = 2 };
bool g5a(enum e5 x) {
  return (unsigned int)x > 3;
}
// CHECK-LABEL: define{{.*}} i1 @g5a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g5a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g5b(enum e5 x) {
  return (unsigned int)x >= 3;
}
// CHECK-LABEL: define{{.*}} i1 @g5b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g5b
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e6 { e6_a = -2 };
bool g6a(enum e6 x) {
  return (int)x > 1;
}
// CHECK-LABEL: define{{.*}} i1 @g6a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g6a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g6b(enum e6 x) {
  return (int)x >= 1;
}
// CHECK-LABEL: define{{.*}} i1 @g6b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g6b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g6c(enum e6 x) {
  return (int)x < -2;
}
// CHECK-LABEL: define{{.*}} i1 @g6c
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g6c
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g6d(enum e6 x) {
  return (int)x <= -2;
}
// CHECK-LABEL: define{{.*}} i1 @g6d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g6d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e7 { e7_a = -1, e7_b = 2 };
bool g7a(enum e7 x) {
  return (int)x > 3;
}
// CHECK-LABEL: define{{.*}} i1 @g7a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g7a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g7b(enum e7 x) {
  return (int)x >= 3;
}
// CHECK-LABEL: define{{.*}} i1 @g7b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g7b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g7c(enum e7 x) {
  return (int)x < -4;
}
// CHECK-LABEL: define{{.*}} i1 @g7c
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g7c
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g7d(enum e7 x) {
  return (int)x <= -4;
}
// CHECK-LABEL: define{{.*}} i1 @g7d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g7d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e8 { e8_a = (unsigned int)-1 };
bool g8b(enum e8 x) {
  return (unsigned int)x >= (unsigned int)-1;
}
// CHECK-LABEL: define{{.*}} i1 @g8b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g8b
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e9 { e9_a = (unsigned int)-2 };
bool g9b(enum e9 x) {
  return (unsigned int)x >= (unsigned int)-1;
}
// CHECK-LABEL: define{{.*}} i1 @g9b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g9b
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e10 { e10_a = (int)(((unsigned int)-1) >> 1) };
bool g10a(enum e10 x) {
  return (unsigned int)x > (int)(((unsigned int)-1) >> 1);
}
// CHECK-LABEL: define{{.*}} i1 @g10a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g10a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g10b(enum e10 x) {
  return (unsigned int)x >= (int)(((unsigned int)-1) >> 1);
}
// CHECK-LABEL: define{{.*}} i1 @g10b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g10b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g10d(enum e10 x) {
  return (unsigned int)x <= 0;
}
// CHECK-LABEL: define{{.*}} i1 @g10d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g10d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e11 { e11_a = -(int)(((unsigned int)-1) >> 1) };
bool g11b(enum e11 x) {
  return (int)x >= (int)(((unsigned int)-1) >> 1);
}
// CHECK-LABEL: define{{.*}} i1 @g11b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g11b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g11d(enum e11 x) {
  return (int)x <= -(int)(((unsigned int)-1) >> 1) - 1;
}
// CHECK-LABEL: define{{.*}} i1 @g11d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g11d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e12 { e12_a = -(long long)(((unsigned long long)-1) >> 2) };
bool g12a(enum e12 x) {
  return (long long)x > (long long)(((unsigned long long)-1) >> 2);
}
// CHECK-LABEL: define{{.*}} i1 @g12a
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g12a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g12b(enum e12 x) {
  return (long long)x >= (long long)(((unsigned long long)-1) >> 2);
}
// CHECK-LABEL: define{{.*}} i1 @g12b
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g12b
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g12c(enum e12 x) {
  return (long long)x < -(long long)(((unsigned long long)-1) >> 2) - 1;
}
// CHECK-LABEL: define{{.*}} i1 @g12c
// CHECK: ret i1 false
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g12c
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g12d(enum e12 x) {
  return (long long)x <= -(long long)(((unsigned long long)-1) >> 2) - 1;
}
// CHECK-LABEL: define{{.*}} i1 @g12d
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g12d
// NO-STRICT-ENUMS-C: ret i1 %cmp

enum e13 : long long { e13_a = -(long long)(((unsigned long long)-1) >> 2) };
bool g13a(enum e13 x) {
  return (long long)x > (long long)(((unsigned long long)-1) >> 2);
}
// CHECK-LABEL: define{{.*}} i1 @g13a
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g13a
// NO-STRICT-ENUMS-C: ret i1 %cmp
bool g13c(enum e13 x) {
  return (long long)x < -(long long)(((unsigned long long)-1) >> 2) - 1;
}
// CHECK-LABEL: define{{.*}} i1 @g13c
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-C-LABEL: define{{.*}} i1 @g13c
// NO-STRICT-ENUMS-C: ret i1 %cmp

#ifdef __cplusplus
enum class e14 { e14_a = -(int)(((unsigned int)-1) >> 2) };
bool g14a(enum e14 x) {
  return (int)x > (int)(((unsigned int)-1) >> 2);
}
// CHECK-LABEL: define{{.*}} i1 @g14a
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-LABEL: define{{.*}} i1 @g14a
// NO-STRICT-ENUMS: ret i1 %cmp
bool g14c(enum e14 x) {
  return (int)x < -(int)(((unsigned int)-1) >> 2) - 1;
}
// CHECK-LABEL: define{{.*}} i1 @g14c
// CHECK: ret i1 %cmp
// NO-STRICT-ENUMS-LABEL: define{{.*}} i1 @g14c
// NO-STRICT-ENUMS: ret i1 %cmp
#endif // __cplusplus

#ifdef __cplusplus
}
#endif // __cplusplus
