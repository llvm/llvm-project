// RUN: %clang_cc1 %s -emit-llvm -o - -fblocks -triple x86_64-apple-darwin10 | FileCheck %s

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
// CHECK:  [[RP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[C1]], i32 0, i32 0
// CHECK-NEXT:  [[R:%.*]] = load double, ptr [[RP]]
// CHECK-NEXT:  [[IP:%.*]] = getelementptr inbounds nuw { double, double }, ptr [[C1]], i32 0, i32 1
// CHECK-NEXT:  [[I:%.*]] = load double, ptr [[IP]]
