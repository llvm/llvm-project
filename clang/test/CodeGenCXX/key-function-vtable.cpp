// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-none-linux-gnu %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers -triple arm-apple-darwin %s -emit-llvm -o - | FileCheck %s

// Simple key function test
struct testa { virtual void a(); };
void testa::a() {}

// Simple key function test
struct testb { virtual void a() {} };
testb *testbvar = new testb;

// Key function with out-of-line inline definition
struct testc { virtual void a(); };
inline void testc::a() {}

// Functions with inline specifier are not key functions (PR5705)
struct testd { inline virtual void a(); };
void testd::a() {}

// Functions with inline specifier are not key functions (PR5705)
struct teste { inline virtual void a(); };
teste *testevar = new teste;

// Key functions with namespace (PR5711)
namespace {
  struct testf { virtual void a(); };
}
void testf::a() {}

// Key functions with namespace (PR5711)
namespace {
  struct testg { virtual void a(); };
}
void testg::a() {}
testg *testgvar = new testg;

struct X0 { virtual ~X0(); };
struct X1 : X0 {
  virtual void f();
};

inline void X1::f() { }

void use_X1() { X1 x1; }

// CHECK-DAG: @_ZTV2X1 = linkonce_odr unnamed_addr constant
// CHECK-DAG: @_ZTV5testa ={{.*}} unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null
// CHECK-DAG: @_ZTV5testc = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null
// CHECK-DAG: @_ZTV5testb = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null
// CHECK-DAG: @_ZTV5teste = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null
// CHECK-DAG: @_ZTVN12_GLOBAL__N_15testgE = internal unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null
