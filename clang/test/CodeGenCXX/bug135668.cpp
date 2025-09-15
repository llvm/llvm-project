// RUN: %clang_cc1 %s -triple arm64-apple-macosx -emit-llvm -fcxx-exceptions -fexceptions -std=c++23 -o - | FileCheck  %s

class TestClass {
  public:
   TestClass();
   int field = 0;
   friend class Foo;
   static void * operator new(unsigned long size);
  private:
   static void operator delete(void *p);
 };

class Foo {
public:
  int test_method();
};

int Foo::test_method() {
  TestClass *obj = new TestClass() ;
  return obj->field;
}

// CHECK-LABEL: define noundef i32 @_ZN3Foo11test_methodEv
// CHECK: [[THIS_ADDR:%.*]] = alloca ptr, align 8
// CHECK: [[OBJ:%.*]] = alloca ptr, align 8
// CHECK: store ptr %this, ptr [[THIS_ADDR]], align 8
// CHECK: [[THIS1:%.*]] = load ptr, ptr [[THIS_ADDR]], align 8
// CHECK: [[ALLOCATION:%.*]] = call noundef ptr @_ZN9TestClassnwEm(i64 noundef 4)
// CHECK: [[INITIALIZEDOBJ:%.*]] = invoke noundef ptr @_ZN9TestClassC1Ev(ptr noundef nonnull align 4 dereferenceable(4) [[ALLOCATION]])
// CHECK-NEXT:  to label %[[INVOKE_CONT:.*]] unwind label %[[LPAD:.*]]
// CHECK: [[INVOKE_CONT]]:
// CHECK: store ptr [[ALLOCATION]], ptr [[OBJ]], align 8
// CHECK: [[OBJPTR:%.*]] = load ptr, ptr [[OBJ]], align 8
// CHECK: [[FIELDPTR:%.*]] = getelementptr inbounds nuw %class.TestClass, ptr [[OBJPTR]], i32 0, i32 0
// CHECK: [[FIELD:%.*]] = load i32, ptr [[FIELDPTR]], align 4
// CHECK: ret i32 [[FIELD]]
// CHECK: [[LPAD]]:
// CHECK: call void @_ZN9TestClassdlEPv(ptr noundef [[ALLOCATION]]) #3
