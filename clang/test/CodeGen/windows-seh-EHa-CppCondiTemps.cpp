// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local noundef i32 @"?bar@@YAHHVB1@@VB2@@@Z"
// CHECK: %coerce.dive1 = getelementptr inbounds nuw %class.B2
// CHECK: %coerce.dive2 = getelementptr inbounds nuw %class.B1
// -----   scope begin of two passed-by-value temps  
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B1@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B2@@QEAA@XZ"

// CHECK: define linkonce_odr dso_local void @"??1B2@@QEAA@XZ"
// CHECK: %this.addr = alloca ptr
// -----  B1 scope begin without base ctor
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B1@@QEAA@XZ"

// CHECK: define dso_local void @"?goo@@YA?AVB1@@H@Z"
// CHECK: call noundef ptr @"??0B2@@QEAA@XZ"(ptr
// CHECK: invoke void @llvm.seh.scope.begin()
// check: call void @llvm.memcpy
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B2@@QEAA@XZ"(ptr

// CHECK: define linkonce_odr dso_local noundef ptr @"??0B2@@QEAA@XZ"
// CHECK: call noundef ptr @"??0B1@@QEAA@XZ"(ptr
// -----  scope begin of base ctor 
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()
// -----  B1 scope end without base dtor

// ****************************************************************************
// Abstract:     Test CPP Conditional-Expr & ABI Temps under SEH -EHa option

void printf(...);

int xxxx = 0;
int* ptr; 

int foo(int a)
{
  return xxxx + a;
}

class B1      {
public:
   int data = 90;
   B1() { foo(data + 111); }
    ~B1() { printf("in B1 Dtor \n"); }
};
class B2 : public B1 {
public:
  B2() { foo(data + 222); }
  ~B2() { printf("in B2 Dtor \n");; }
};
class B3 : public B2 {
public:
  B3() { foo(data + 333); }
  ~B3() { printf("in B3 Dtor \n");; }
};

int bar(int j, class B1 b1Bar, class B2 b2Bar)
{
  int ww;
  if ( j > 0)
    ww = b1Bar.data;
  else
    ww = b2Bar.data;
  return  ww + *ptr;
}

class B1 goo(int w)
{
  class B2 b2ingoo;
  b2ingoo.data += w;
  return b2ingoo;
}

// CHECK: define dso_local noundef i32 @main()
// CHECK: invoke void @llvm.seh.scope.begin()
// ---   beginning of conditional temp test
// CHECK: invoke noundef ptr @"??0B2@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke noundef ptr @"??0B3@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B3@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: call void @"??1B2@@QEAA@XZ"
// -----  end of conditional temp test

// -----  testing caller's passed-by-value temps
//        setting scope in case exception occurs before the call
// check: invoke ptr @"??0B2@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke noundef ptr @"??0B1@@QEAA@XZ"
// CHECK: invoke void @llvm.seh.scope.begin()
// -----   end of temps' scope right before callee
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: invoke void @llvm.seh.scope.end()
// CHECK: invoke noundef i32 @"?bar@@YAHHVB1@@VB2@@@Z"

// -----  testing caller's return-by-value temp
//        scope begins right after callee which is the ctor of return temp
// CHECK: void @"?goo@@YA?AVB1@@H@Z"
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()

int main() {
  class B3 b3inmain;

  // Test conditional ctor and dtor
  int m = (xxxx > 1) ? B2().data + foo(99) : 
      B3().data + foo(88);

  // Test: passed-in by value
  // Per Windows ABI, ctored by caller, dtored by callee
  int i = bar(foo(0), B1(), B2());

  // Test: returned by value
  // Per Windows ABI, caller allocate a temp in stack, then ctored by callee, 
  //          finally dtored in caller after consumed
  class B1 b1fromgoo = goo(i);

  return m + b1fromgoo.data + b3inmain.data;
}
