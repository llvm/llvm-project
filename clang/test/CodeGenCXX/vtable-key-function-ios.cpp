// RUN: %clang_cc1 %s -triple=armv7-apple-darwin -emit-llvm -o - | FileCheck -check-prefixes=CHECK,CHECK-UNIX %s
// RUN: %clang_cc1 %s -triple=armv7-apple-darwin -emit-llvm -o - | FileCheck -check-prefix=CHECK-LATE %s

// RUN: %clang_cc1 %s -triple=x86_64-pc-windows-gnu -emit-llvm -o - | FileCheck -check-prefixes=CHECK,CHECK-MINGW %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-windows-gnu -emit-llvm -o - | FileCheck -check-prefix=CHECK-LATE %s

// The 'a' variants ask for the vtable first.
// The 'b' variants ask for the vtable second.
// The 'c' variants ask for the vtable third.
// We do a separate CHECK-LATE pass because the RTTI definition gets
// changed after the fact, which causes reordering of the globals.

// These are not separated into namespaces because the way that Sema
// currently reports namespaces to IR-generation (i.e., en masse for
// the entire namespace at once) subverts the ordering that we're
// trying to test.

namespace std { class type_info; }
extern void use(const std::type_info &rtti);

/*** Test0a ******************************************************************/

struct Test0a {
  Test0a();
  virtual inline void foo();
  virtual void bar();
};

// V-table should be defined externally.
Test0a::Test0a() { use(typeid(Test0a)); }
// CHECK: @_ZTV6Test0a = external {{(dso_local )?}}unnamed_addr constant 
// CHECK-UNIX: @_ZTI6Test0a = external {{(dso_local )?}}constant
// CHECK-MINGW: @_ZTI6Test0a = linkonce_odr {{(dso_local )?}}constant

// This is not a key function.
void Test0a::foo() {}

/*** Test0b ******************************************************************/

struct Test0b {
  Test0b();
  virtual inline void foo();
  virtual void bar();
};

// This is not a key function.
void Test0b::foo() {}

// V-table should be defined externally.
Test0b::Test0b() { use(typeid(Test0b)); }
// CHECK: @_ZTV6Test0b = external {{(dso_local )?}}unnamed_addr constant 
// CHECK-UNIX: @_ZTI6Test0b = external {{(dso_local )?}}constant
// CHECK-MINGW: @_ZTI6Test0b = linkonce_odr {{(dso_local )?}}constant

/*** Test1a ******************************************************************/

struct Test1a {
  Test1a();
  virtual void foo();
  virtual void bar();
};

// V-table needs to be defined weakly.
Test1a::Test1a() { use(typeid(Test1a)); }
// CHECK:      @_ZTV6Test1a = linkonce_odr {{(dso_local )?}}unnamed_addr constant 
// CHECK-LATE: @_ZTS6Test1a = linkonce_odr {{(dso_local )?}}constant
// CHECK-LATE: @_ZTI6Test1a = linkonce_odr {{(dso_local )?}}constant

// This defines the key function.
inline void Test1a::foo() {}

/*** Test1b ******************************************************************/

struct Test1b {
  Test1b();
  virtual void foo();
  virtual void bar();
};

// This defines the key function.
inline void Test1b::foo() {}

// V-table should be defined weakly..
Test1b::Test1b() { use(typeid(Test1b)); }
// CHECK: @_ZTV6Test1b = linkonce_odr {{(dso_local )?}}unnamed_addr constant 
// CHECK: @_ZTS6Test1b = linkonce_odr {{(dso_local )?}}constant
// CHECK: @_ZTI6Test1b = linkonce_odr {{(dso_local )?}}constant

/*** Test2a ******************************************************************/

struct Test2a {
  Test2a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined with weak linkage.
Test2a::Test2a() { use(typeid(Test2a)); }
// CHECK:      @_ZTV6Test2a = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK-LATE: @_ZTS6Test2a = linkonce_odr {{(dso_local )?}}constant
// CHECK-LATE: @_ZTI6Test2a = linkonce_odr {{(dso_local )?}}constant

void Test2a::bar() {}
inline void Test2a::foo() {}

/*** Test2b ******************************************************************/

struct Test2b {
  Test2b();
  virtual void foo();
  virtual void bar();
};

void Test2b::bar() {}

// V-table should be defined with weak linkage.
Test2b::Test2b() { use(typeid(Test2b)); }
// CHECK:      @_ZTV6Test2b = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK-LATE: @_ZTS6Test2b = linkonce_odr {{(dso_local )?}}constant
// CHECK-LATE: @_ZTI6Test2b = linkonce_odr {{(dso_local )?}}constant

inline void Test2b::foo() {}

/*** Test2c ******************************************************************/

struct Test2c {
  Test2c();
  virtual void foo();
  virtual void bar();
};

void Test2c::bar() {}
inline void Test2c::foo() {}

// V-table should be defined with weak linkage.
Test2c::Test2c() { use(typeid(Test2c)); }
// CHECK: @_ZTV6Test2c = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK: @_ZTS6Test2c = linkonce_odr {{(dso_local )?}}constant
// CHECK: @_ZTI6Test2c = linkonce_odr {{(dso_local )?}}constant

/*** Test3a ******************************************************************/

struct Test3a {
  Test3a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined with weak linkage.
Test3a::Test3a() { use(typeid(Test3a)); }
// CHECK:      @_ZTV6Test3a = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK-LATE: @_ZTS6Test3a = linkonce_odr {{(dso_local )?}}constant
// CHECK-LATE: @_ZTI6Test3a = linkonce_odr {{(dso_local )?}}constant

// This defines the key function.
inline void Test3a::bar() {}
inline void Test3a::foo() {}

/*** Test3b ******************************************************************/

struct Test3b {
  Test3b();
  virtual void foo();
  virtual void bar();
};

inline void Test3b::bar() {}

// V-table should be defined with weak linkage.
Test3b::Test3b() { use(typeid(Test3b)); }
// CHECK:      @_ZTV6Test3b = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK-LATE: @_ZTS6Test3b = linkonce_odr {{(dso_local )?}}constant
// CHECK-LATE: @_ZTI6Test3b = linkonce_odr {{(dso_local )?}}constant

// This defines the key function.
inline void Test3b::foo() {}

/*** Test3c ******************************************************************/

struct Test3c {
  Test3c();
  virtual void foo();
  virtual void bar();
};

// This defines the key function.
inline void Test3c::bar() {}
inline void Test3c::foo() {}

// V-table should be defined with weak linkage.
Test3c::Test3c() { use(typeid(Test3c)); }
// CHECK: @_ZTV6Test3c = linkonce_odr {{(dso_local )?}}unnamed_addr constant
// CHECK: @_ZTS6Test3c = linkonce_odr {{(dso_local )?}}constant
// CHECK: @_ZTI6Test3c = linkonce_odr {{(dso_local )?}}constant
