// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// See Test9 for test description.
// CHECK: @_ZTTN5Test91BE = linkonce_odr unnamed_addr constant
namespace Test1 {

// Check that we don't initialize the vtable pointer in A::~A(), since the destructor body is trivial.
struct A {
  virtual void f();
  ~A();
};

// CHECK-LABEL: define void @_ZN5Test11AD2Ev
// CHECK-NOT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test11AE, i64 0, i64 2) to i32 (...)**), i32 (...)***
A::~A() 
{
}

}

namespace Test2 {

// Check that we do initialize the vtable pointer in A::~A() since the destructor body isn't trivial.
struct A {
  virtual void f();
  ~A();
};

// CHECK-LABEL: define void @_ZN5Test21AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test21AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A() {
  f();
}

}

namespace Test3 {

// Check that we don't initialize the vtable pointer in A::~A(), since the destructor body is trivial
// and Field's destructor body is also trivial.
struct Field {
  ~Field() { }
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test31AD2Ev
// CHECK-NOT: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test31AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A() {
  
}

}

namespace Test4 {

// Check that we do initialize the vtable pointer in A::~A(), since Field's destructor body
// isn't trivial.

void f();

struct Field {
  ~Field() { f(); }
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test41AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test41AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A()
{
}

}

namespace Test5 {

// Check that we do initialize the vtable pointer in A::~A(), since Field's destructor isn't
// available in this translation unit.

struct Field {
  ~Field();
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test51AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test51AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A()
{
}

}

namespace Test6 {

// Check that we do initialize the vtable pointer in A::~A(), since Field has a member
// variable with a non-trivial destructor body.

struct NonTrivialDestructorBody {
  ~NonTrivialDestructorBody();
};

struct Field {
  NonTrivialDestructorBody nonTrivialDestructorBody;
};

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test61AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test61AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A()
{
}

}

namespace Test7 {

// Check that we do initialize the vtable pointer in A::~A(), since Field has a base
// class with a non-trivial destructor body.

struct NonTrivialDestructorBody {
  ~NonTrivialDestructorBody();
};

struct Field : NonTrivialDestructorBody { };

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test71AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test71AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A()
{
}

}

namespace Test8 {

// Check that we do initialize the vtable pointer in A::~A(), since Field has a virtual base
// class with a non-trivial destructor body.

struct NonTrivialDestructorBody {
  ~NonTrivialDestructorBody();
};

struct Field : virtual NonTrivialDestructorBody { };

struct A {
  virtual void f();
  ~A();

  Field field;
};

// CHECK-LABEL: define void @_ZN5Test81AD2Ev
// CHECK: store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN5Test81AE, i32 0, inrange i32 0, i32 2) to i32 (...)**), i32 (...)***
A::~A()
{
}

}

namespace Test9 {

// Check that we emit a VTT for B, even though we don't initialize the vtable pointer in the destructor.
struct A { virtual ~A () { } };
struct B : virtual A {};
struct C : virtual B { 
  virtual ~C();
};
C::~C() {}

}
