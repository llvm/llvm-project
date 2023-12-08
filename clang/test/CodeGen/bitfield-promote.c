// RUN: %clang -O3 -emit-llvm -S -o - %s | FileCheck %s

long long f0(void) {
 struct { unsigned f0 : 32; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}
// CHECK: @f0()
// CHECK: ret i64 4294967292

long long f1(void) {
 struct { unsigned f0 : 31; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}
// CHECK: @f1()
// CHECK: ret i64 -4

long long f2(void) {
 struct { unsigned f0     ; } x = { 18 };
 return (long long) (x.f0 - (int) 22);
}
// CHECK: @f2()
// CHECK: ret i64 4294967292
