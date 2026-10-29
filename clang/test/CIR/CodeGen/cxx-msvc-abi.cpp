// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// Adapted from Clang CodeGenCXX/microsoft-abi-methods.cpp,
// microsoft-abi-structors.cpp, and microsoft-abi-vftables.cpp.

//===----------------------------------------------------------------------===//
// Section 1: Member Functions & Calling Conventions
//===----------------------------------------------------------------------===//

class MethodsTest {
public:
  void simple_method() {}
  void vararg_method(const char *fmt, ...) {}
  static void static_method() {}
};

void call_methods() {
  MethodsTest obj;
  obj.simple_method();
  obj.vararg_method("test");
  MethodsTest::static_method();
}

// CIR-DAG: cir.global {{.*}} linkonce_odr {{.*}} @"??_C@_
// CIR-DAG: cir.func {{.*}} @"?simple_method@MethodsTest@@QEAAXXZ"(%{{.*}}: !cir.ptr<!rec_MethodsTest>
// CIR-DAG: cir.func {{.*}} @"?vararg_method@MethodsTest@@QEAAXPEBDZZ"(%{{.*}}: !cir.ptr<!rec_MethodsTest>{{.*}}, %{{.*}}: !cir.ptr<!s8i>{{.*}}, ...)
// CIR-DAG: cir.func {{.*}} @"?static_method@MethodsTest@@SAXXZ"()

//===----------------------------------------------------------------------===//
// Section 2: Structors and 'this' Return Calling Convention
//===----------------------------------------------------------------------===//

class BasicA {
  int x;
public:
  BasicA() { x = 42; }
};

void test_basic() {
  BasicA a;
}

// In the MSVC ABI, constructors return 'this' (hasThisReturn is true).
// CIR-LABEL: cir.func {{.*}} @"??0BasicA@@QEAA@XZ"
// CIR-SAME: (%[[THIS_ARG:.*]]: !cir.ptr<!rec_BasicA>{{.*}}) -> (!cir.ptr<!rec_BasicA>
// CIR:   %[[RETVAL:.*]] = cir.alloca "__retval"
// CIR:   %[[THIS_ALLOCA:.*]] = cir.alloca "this"
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ALLOCA]]
// CIR:   %[[LOADED_THIS:.*]] = cir.load %[[THIS_ALLOCA]]
// CIR:   cir.store {{.*}}%[[LOADED_THIS]], %[[RETVAL]]
// CIR:   cir.const #cir.int<42>
// CIR:   %[[LOADED_RET:.*]] = cir.load %[[RETVAL]]
// CIR:   cir.return %[[LOADED_RET]] : !cir.ptr<!rec_BasicA>
// CIR: }

// Constructor & Destructor Inheritance chain:
class Base {
public:
  Base();
};

class Derived : public Base {
public:
  Derived();
};

Derived::Derived() {}

// Derived ctor calls Base ctor and returns this:
// CIR-LABEL: cir.func {{.*}} @"??0Derived@@QEAA@XZ"
// CIR-SAME: (%[[THIS:.*]]: !cir.ptr<!rec_Derived>{{.*}}) -> (!cir.ptr<!rec_Derived>
// CIR:   cir.call @"??0Base@@QEAA@XZ"
// CIR:   cir.return %{{.*}} : !cir.ptr<!rec_Derived>
