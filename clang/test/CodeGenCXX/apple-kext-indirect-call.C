// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -emit-llvm -o - %s | FileCheck %s

struct Base { 
  virtual void abc(void) const; 
};

void Base::abc(void) const {}

void FUNC(Base* p) {
  p->Base::abc();
}

// CHECK: getelementptr inbounds (void (%struct.Base*)** bitcast ([3 x i8*]* @_ZTV4Base to void (%struct.Base*)**), i64 2)
// CHECK-NOT: call void @_ZNK4Base3abcEv
