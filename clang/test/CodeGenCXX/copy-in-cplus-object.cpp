// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

struct S {
  S(const char *);
  ~S();
};

struct TestObject
{
	TestObject(const TestObject& inObj, int def = 100,  const S &Silly = "silly");
	TestObject();
	~TestObject();
	TestObject& operator=(const TestObject& inObj);
	int version() const;

};

void testRoutine() {
    TestObject one;
    int (^V)() = ^{ return one.version(); };
}

// CHECK: call void @_ZN10TestObjectC1Ev
// CHECK: call void @_ZN1SC1EPKc
// CHECK: call void @_ZN10TestObjectC1ERKS_iRK1S
// CHECK: call void @_ZN1SD1Ev
// CHECK: call void @_ZN10TestObjectD1Ev
// CHECK: call void @_ZN10TestObjectD1Ev
