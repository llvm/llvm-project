// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fapple-kext -fno-rtti -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV1A = unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast ({ i8*, i32, i64, i64 }* @_ZNK1A3abcEv.ptrauth to i8*), i8* null] }
// CHECK: @_ZTV4Base = unnamed_addr constant { [4 x i8*] } { [4 x i8*] [i8* null, i8* null, i8* bitcast ({ i8*, i32, i64, i64 }* @_ZNK4Base3abcEv.ptrauth to i8*), i8* null] }
// CHECK: @_ZTV8Derived2 = unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast ({ i8*, i32, i64, i64 }* @_ZNK8Derived23efgEv.ptrauth to i8*), i8* null] }
// CHECK: @_ZTV2D2 = unnamed_addr constant { [5 x i8*] } { [5 x i8*] [i8* null, i8* null, i8* null, i8* bitcast ({ i8*, i32, i64, i64 }* @_ZNK2D23abcEv.ptrauth to i8*), i8* null] }

struct A {
  virtual const char* abc(void) const;
};

const char* A::abc(void) const {return "A"; };

struct B : virtual A {
  virtual void VF();
};

void B::VF() {}

void FUNC(B* p) {
// CHECK: [[T1:%.*]] = load i8* (%struct.A*)*, i8* (%struct.A*)** getelementptr inbounds (i8* (%struct.A*)*, i8* (%struct.A*)** bitcast ({ [4 x i8*] }* @_ZTV1A to i8* (%struct.A*)**), i64 2)
// CHECK-NEXT:  [[BT1:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i8* (%struct.A*)** getelementptr inbounds (i8* (%struct.A*)*, i8* (%struct.A*)** bitcast ({ [4 x i8*] }* @_ZTV1A to i8* (%struct.A*)**), i64 2) to i64), i64 12401)
// CHECK-NEXT:  [[T2:%.*]] = call i8* [[T1]](%struct.A* {{.*}}) [ "ptrauth"(i32 0, i64 [[BT1]]) ]
  const char* c = p->A::abc();
}


// Test2
struct Base { virtual char* abc(void) const; };

char* Base::abc() const { return 0; }

struct Derived : public Base {
};

void FUNC1(Derived* p) {
// CHECK: [[U1:%.*]] = load i8* (%struct.Base*)*, i8* (%struct.Base*)** getelementptr inbounds (i8* (%struct.Base*)*, i8* (%struct.Base*)** bitcast ({ [4 x i8*] }* @_ZTV4Base to i8* (%struct.Base*)**), i64 2)
// CHECK-NEXT: [[BU1:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i8* (%struct.Base*)** getelementptr inbounds (i8* (%struct.Base*)*, i8* (%struct.Base*)** bitcast ({ [4 x i8*] }* @_ZTV4Base to i8* (%struct.Base*)**), i64 2) to i64), i64 64320)
// CHECK-NEXT:  [[U2:%.*]] = call i8* [[U1]](%struct.Base* {{.*}}) [ "ptrauth"(i32 0, i64 [[BU1]]) ]
  char* c = p->Base::abc();
}


// Test3
struct Base2 { };

struct Derived2 : virtual Base2 {
  virtual char* efg(void) const;
};

char* Derived2::efg(void) const { return 0; }

void FUNC2(Derived2* p) {
// CHECK: [[V1:%.*]] = load i8* (%struct.Derived2*)*, i8* (%struct.Derived2*)** getelementptr inbounds (i8* (%struct.Derived2*)*, i8* (%struct.Derived2*)** bitcast ({ [5 x i8*] }* @_ZTV8Derived2 to i8* (%struct.Derived2*)**), i64 3)
// CHECK-NEXT:  [[BV1:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i8* (%struct.Derived2*)** getelementptr inbounds (i8* (%struct.Derived2*)*, i8* (%struct.Derived2*)** bitcast ({ [5 x i8*] }* @_ZTV8Derived2 to i8* (%struct.Derived2*)**), i64 3) to i64), i64 36603)
// CHECK-NEXT:  [[V2:%.*]] = call i8* [[V1]](%struct.Derived2* {{.*}}) [ "ptrauth"(i32 0, i64 [[BV1]]) ]
  char* c = p->Derived2::efg();
}

// Test4
struct Base3 { };

struct D1 : virtual Base3 {
};

struct D2 : virtual Base3 {
 virtual char *abc(void) const;
};

struct Sub : D1, D2 {
};

char* D2::abc(void) const { return 0; }

void FUNC3(Sub* p) {
// CHECK: [[W1:%.*]] = load i8* (%struct.D2*)*, i8* (%struct.D2*)** getelementptr inbounds (i8* (%struct.D2*)*, i8* (%struct.D2*)** bitcast ({ [5 x i8*] }* @_ZTV2D2 to i8* (%struct.D2*)**), i64 3)
// CHECK-NEXT:  [[BW1:%.*]] = call i64 @llvm.ptrauth.blend.i64(i64 ptrtoint (i8* (%struct.D2*)** getelementptr inbounds (i8* (%struct.D2*)*, i8* (%struct.D2*)** bitcast ({ [5 x i8*] }* @_ZTV2D2 to i8* (%struct.D2*)**), i64 3) to i64), i64 20222)
// CHECK-NEXT:  [[W2:%.*]] = call i8* [[W1]](%struct.D2* {{.*}}) [ "ptrauth"(i32 0, i64 [[BW1]]) ]
  char* c = p->D2::abc();
}


// Test4
struct Base4 { virtual void abc(); };

void Base4::abc() {}

struct Derived4 : public Base4 {
  void abc() override;
};

void Derived4::abc() {}

void FUNC4(Derived4* p) {
// CHECK: %[[VTABLE:[a-z]+]] = load void (%struct.Derived4*)**, void (%struct.Derived4*)*** %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint void (%struct.Derived4*)** %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth.i64(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to void (%struct.Derived4*)**
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds void (%struct.Derived4*)*, void (%struct.Derived4*)** %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load void (%struct.Derived4*)*, void (%struct.Derived4*)** %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint void (%struct.Derived4*)** %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend.i64(i64 %[[T6]], i64 426)
// CHECK: call void %[[T5]](%struct.Derived4* nonnull dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]
  p->abc();
}
