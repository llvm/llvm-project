// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -Wmicrosoft %s -emit-llvm -o - | FileCheck %s

class Test1 {
public:
   int a;
};

void f1() {
  Test1 var;
  var.Test1::Test1();

  // CHECK:   call void @llvm.memcpy.p0i8.p0i8.i32(i8* %{{.*}}, i8* %{{.*}}, i32 4, i32 4, i1 false)
  var.Test1::Test1(var);
}

class Test2 {
public:
  Test2() { a = 10; b = 10; }
   int a;
   int b;
};

void f2() {
  // CHECK:  %var = alloca %class.Test2, align 4
  // CHECK-NEXT:  call void @_ZN5Test2C1Ev(%class.Test2* %var)
  Test2 var;

  // CHECK-NEXT:  call void @_ZN5Test2C1Ev(%class.Test2* %var)
  var.Test2::Test2();

  // CHECK:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %{{.*}}, i8* %{{.*}}, i32 8, i32 4, i1 false)
  var.Test2::Test2(var);
}




class Test3 {
public:
  Test3() { a = 10; b = 15; c = 20; }
  Test3(const Test3& that) { a = that.a; b = that.b; c = that.c; }
   int a;
   int b;
   int c;
};

void f3() {
  // CHECK: call void @_ZN5Test3C1Ev(%class.Test3* %var)
  Test3 var;

  // CHECK-NEXT: call void @_ZN5Test3C1Ev(%class.Test3* %var2)
  Test3 var2;

  // CHECK-NEXT: call void @_ZN5Test3C1Ev(%class.Test3* %var)
  var.Test3::Test3();

  // CHECK-NEXT: call void @_ZN5Test3C1ERKS_(%class.Test3* %var, %class.Test3* %var2)
  var.Test3::Test3(var2);
}

