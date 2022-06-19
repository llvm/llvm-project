// RUN: %clang_cc1 -no-opaque-pointers %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s
// rdar://10033986

typedef void (^BLOCK)(void);
int main (void)
{
    _Complex double c;
    BLOCK b =  ^() {
      _Complex double z;
      z = z + c;
    };
    b();
}

// CHECK-LABEL: define internal void @__main_block_invoke
// CHECK:  [[C1:%.*]] = alloca { double, double }, align 8
// CHECK:  [[RP:%.*]] = getelementptr inbounds { double, double }, { double, double }* [[C1]], i32 0, i32 0
// CHECK-NEXT:  [[R:%.*]] = load double, double* [[RP]]
// CHECK-NEXT:  [[IP:%.*]] = getelementptr inbounds { double, double }, { double, double }* [[C1]], i32 0, i32 1
// CHECK-NEXT:  [[I:%.*]] = load double, double* [[IP]]
