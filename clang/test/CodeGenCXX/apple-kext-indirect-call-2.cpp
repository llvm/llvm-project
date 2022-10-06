// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -fno-rtti -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZTV1A ={{.*}} unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZNK1A3abcEv, ptr null] }
// CHECK: @_ZTV4Base ={{.*}} unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr null, ptr @_ZNK4Base3abcEv, ptr null] }
// CHECK: @_ZTV8Derived2 ={{.*}} unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr @_ZNK8Derived23efgEv, ptr null] }
// CHECK: @_ZTV2D2 ={{.*}} unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr @_ZNK2D23abcEv, ptr null] }

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
// CHECK-NEXT:  [[T2:%.*]] = call noundef ptr [[T1]]
  const char* c = p->A::abc();
}


// Test2
struct Base { virtual char* abc(void) const; };

char* Base::abc() const { return 0; }

struct Derived : public Base {
};

void FUNC1(Derived* p) {
// CHECK: [[U1:%.*]] = load ptr, ptr getelementptr inbounds (ptr, ptr @_ZTV4Base, i64 2)
// CHECK-NEXT:  [[U2:%.*]] = call noundef ptr [[U1]]
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
// CHECK-NEXT:  [[V2:%.*]] = call noundef ptr [[V1]]
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
// CHECK-NEXT:  [[W2:%.*]] = call noundef ptr [[W1]]
  char* c = p->D2::abc();
}

