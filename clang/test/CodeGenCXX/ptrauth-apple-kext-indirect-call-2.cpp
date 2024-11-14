// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fapple-kext -fno-rtti -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV1A = unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZNK1A3abcEv, i32 0, i64 12401, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2)), ptr null] }, align 8
// CHECK: @_ZTV4Base = unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr ptrauth (ptr @_ZNK4Base3abcEv, i32 0, i64 64320, ptr getelementptr inbounds ({ [4 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2)), ptr null] }, align 8
// CHECK: @_ZTV8Derived2 = unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr ptrauth (ptr @_ZNK8Derived23efgEv, i32 0, i64 36603, ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV8Derived2, i32 0, i32 0, i32 3)), ptr null] }, align 8
// CHECK: @_ZTV2D2 = unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr ptrauth (ptr @_ZNK2D23abcEv, i32 0, i64 20222, ptr getelementptr inbounds ({ [5 x ptr] }, ptr @_ZTV2D2, i32 0, i32 0, i32 3)), ptr null] }, align 8

struct A {
  virtual const char* abc(void) const;
};

const char* A::abc(void) const {return "A"; };

struct B : virtual A {
  virtual void VF();
};

void B::VF() {}

void FUNC(B* p) {
// CHECK: [[T1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV1A, i64 2)
// CHECK-NEXT:  [[BT1:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV1A, i64 2) to i64), i64 12401)
// CHECK-NEXT:  [[T2:%.*]] = call noundef ptr [[T1]](ptr noundef {{.*}}) [ "ptrauth"(i32 0, i64 [[BT1]]) ]
  const char* c = p->A::abc();
}


// Test2
struct Base { virtual char* abc(void) const; };

char* Base::abc() const { return 0; }

struct Derived : public Base {
};

void FUNC1(Derived* p) {
// CHECK: [[U1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV4Base, i64 2)
// CHECK-NEXT: [[BU1:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV4Base, i64 2) to i64), i64 64320)
// CHECK-NEXT:  [[U2:%.*]] = call noundef ptr [[U1]](ptr noundef {{.*}}) [ "ptrauth"(i32 0, i64 [[BU1]]) ]
  char* c = p->Base::abc();
}


// Test3
struct Base2 { };

struct Derived2 : virtual Base2 {
  virtual char* efg(void) const;
};

char* Derived2::efg(void) const { return 0; }

void FUNC2(Derived2* p) {
// CHECK: [[V1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV8Derived2, i64 3)
// CHECK-NEXT:  [[BV1:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV8Derived2, i64 3) to i64), i64 36603)
// CHECK-NEXT:  [[V2:%.*]] = call noundef ptr [[V1]](ptr noundef {{.*}}) [ "ptrauth"(i32 0, i64 [[BV1]]) ]
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
// CHECK: [[W1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV2D2, i64 3)
// CHECK-NEXT:  [[BW1:%.*]] = call i64 @llvm.ptrauth.blend(i64 ptrtoint (ptr getelementptr inbounds (ptr, ptr @_ZTV2D2, i64 3) to i64), i64 20222)
// CHECK-NEXT:  [[W2:%.*]] = call noundef ptr [[W1]](ptr noundef {{.*}}) [ "ptrauth"(i32 0, i64 [[BW1]]) ]
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
// CHECK: %[[VTABLE:[a-z]+]] = load ptr, ptr %{{.*}}
// CHECK: %[[T0:[0-9]+]] = ptrtoint ptr %[[VTABLE]] to i64
// CHECK: %[[T3:[0-9]+]] = call i64 @llvm.ptrauth.auth(i64 %[[T0]], i32 2, i64 0)
// CHECK: %[[T4:[0-9]+]] = inttoptr i64 %[[T3]] to ptr
// CHECK: %[[VFN:[a-z]+]] = getelementptr inbounds ptr, ptr %[[T4]], i64 0
// CHECK: %[[T5:[0-9]+]] = load ptr, ptr %[[VFN]]
// CHECK: %[[T6:[0-9]+]] = ptrtoint ptr %[[VFN]] to i64
// CHECK: %[[T7:[0-9]+]] = call i64 @llvm.ptrauth.blend(i64 %[[T6]], i64 426)
// CHECK: call void %[[T5]](ptr noundef nonnull align {{[0-9]+}} dereferenceable(8) %{{.*}}) [ "ptrauth"(i32 0, i64 %[[T7]]) ]
  p->abc();
}
