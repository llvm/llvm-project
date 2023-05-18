// RUN: %clang_cc1 %s -triple=armv7-unknown-unknown -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=armv7-unknown-unknown -emit-llvm -o - | FileCheck -check-prefix=CHECK-LATE %s

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
// CHECK: @_ZTV6Test0a = external unnamed_addr constant 
// CHECK: @_ZTI6Test0a = external constant

// This is still not a key function.
void Test0a::foo() {}

/*** Test0b ******************************************************************/

struct Test0b {
  Test0b();
  virtual inline void foo();
  virtual void bar();
};

// This is still not a key function.
void Test0b::foo() {}

// V-table should be defined externally.
Test0b::Test0b() { use(typeid(Test0b)); }
// CHECK: @_ZTV6Test0b = external unnamed_addr constant 
// CHECK: @_ZTI6Test0b = external constant

/*** Test1a ******************************************************************/

struct Test1a {
  Test1a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined externally.
Test1a::Test1a() { use(typeid(Test1a)); }
// CHECK: @_ZTV6Test1a = external unnamed_addr constant 
// CHECK: @_ZTI6Test1a = external constant

// 'bar' becomes the key function when 'foo' is defined inline.
inline void Test1a::foo() {}

/*** Test1b ******************************************************************/

struct Test1b {
  Test1b();
  virtual void foo();
  virtual void bar();
};

// 'bar' becomes the key function when 'foo' is defined inline.
inline void Test1b::foo() {}

// V-table should be defined externally.
Test1b::Test1b() { use(typeid(Test1b)); }
// CHECK: @_ZTV6Test1b = external unnamed_addr constant 
// CHECK: @_ZTI6Test1b = external constant

/*** Test2a ******************************************************************/

struct Test2a {
  Test2a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined with strong linkage.
Test2a::Test2a() { use(typeid(Test2a)); }
// CHECK:      @_ZTV6Test2a ={{.*}} unnamed_addr constant
// CHECK-LATE: @_ZTS6Test2a ={{.*}} constant
// CHECK-LATE: @_ZTI6Test2a ={{.*}} constant

// 'bar' becomes the key function when 'foo' is defined inline.
void Test2a::bar() {}
inline void Test2a::foo() {}

/*** Test2b ******************************************************************/

struct Test2b {
  Test2b();
  virtual void foo();
  virtual void bar();
};

// 'bar' becomes the key function when 'foo' is defined inline.
void Test2b::bar() {}

// V-table should be defined with strong linkage.
Test2b::Test2b() { use(typeid(Test2b)); }
// CHECK:      @_ZTV6Test2b ={{.*}} unnamed_addr constant
// CHECK-LATE: @_ZTS6Test2b ={{.*}} constant
// CHECK-LATE: @_ZTI6Test2b ={{.*}} constant

inline void Test2b::foo() {}

/*** Test2c ******************************************************************/

struct Test2c {
  Test2c();
  virtual void foo();
  virtual void bar();
};

// 'bar' becomes the key function when 'foo' is defined inline.
void Test2c::bar() {}
inline void Test2c::foo() {}

// V-table should be defined with strong linkage.
Test2c::Test2c() { use(typeid(Test2c)); }
// CHECK: @_ZTV6Test2c ={{.*}} unnamed_addr constant
// CHECK: @_ZTS6Test2c ={{.*}} constant
// CHECK: @_ZTI6Test2c ={{.*}} constant

/*** Test3a ******************************************************************/

struct Test3a {
  Test3a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined with weak linkage.
Test3a::Test3a() { use(typeid(Test3a)); }
// CHECK:      @_ZTV6Test3a = linkonce_odr unnamed_addr constant
// CHECK-LATE: @_ZTS6Test3a = linkonce_odr constant
// CHECK-LATE: @_ZTI6Test3a = linkonce_odr constant

// There ceases to be a key function after these declarations.
inline void Test3a::bar() {}
inline void Test3a::foo() {}

/*** Test3b ******************************************************************/

struct Test3b {
  Test3b();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
inline void Test3b::bar() {}

// V-table should be defined with weak linkage.
Test3b::Test3b() { use(typeid(Test3b)); }
// CHECK:      @_ZTV6Test3b = linkonce_odr unnamed_addr constant
// CHECK-LATE: @_ZTS6Test3b = linkonce_odr constant
// CHECK-LATE: @_ZTI6Test3b = linkonce_odr constant

inline void Test3b::foo() {}

/*** Test3c ******************************************************************/

struct Test3c {
  Test3c();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
inline void Test3c::bar() {}
inline void Test3c::foo() {}

// V-table should be defined with weak linkage.
Test3c::Test3c() { use(typeid(Test3c)); }
// CHECK: @_ZTV6Test3c = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test3c = linkonce_odr constant
// CHECK: @_ZTI6Test3c = linkonce_odr constant

/*** Test4a ******************************************************************/

template <class T> struct Test4a {
  Test4a();
  virtual void foo();
  virtual void bar();
};

// V-table should be defined with weak linkage.
template <> Test4a<int>::Test4a() { use(typeid(Test4a)); }
// CHECK: @_ZTV6Test4aIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test4aIiE = linkonce_odr constant
// CHECK: @_ZTI6Test4aIiE = linkonce_odr constant

// There ceases to be a key function after these declarations.
template <> inline void Test4a<int>::bar() {}
template <> inline void Test4a<int>::foo() {}

/*** Test4b ******************************************************************/

template <class T> struct Test4b {
  Test4b();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
template <> inline void Test4b<int>::bar() {}

// V-table should be defined with weak linkage.
template <> Test4b<int>::Test4b() { use(typeid(Test4b)); }
// CHECK: @_ZTV6Test4bIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test4bIiE = linkonce_odr constant
// CHECK: @_ZTI6Test4bIiE = linkonce_odr constant

template <> inline void Test4b<int>::foo() {}

/*** Test4c ******************************************************************/

template <class T> struct Test4c {
  Test4c();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
template <> inline void Test4c<int>::bar() {}
template <> inline void Test4c<int>::foo() {}

// V-table should be defined with weak linkage.
template <> Test4c<int>::Test4c() { use(typeid(Test4c)); }
// CHECK: @_ZTV6Test4cIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test4cIiE = linkonce_odr constant
// CHECK: @_ZTI6Test4cIiE = linkonce_odr constant

/*** Test5a ******************************************************************/

template <class T> struct Test5a {
  Test5a();
  virtual void foo();
  virtual void bar();
};

template <> inline void Test5a<int>::bar();
template <> inline void Test5a<int>::foo();

// V-table should be defined with weak linkage.
template <> Test5a<int>::Test5a() { use(typeid(Test5a)); }
// CHECK: @_ZTV6Test5aIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test5aIiE = linkonce_odr constant
// CHECK: @_ZTI6Test5aIiE = linkonce_odr constant

// There ceases to be a key function after these declarations.
template <> inline void Test5a<int>::bar() {}
template <> inline void Test5a<int>::foo() {}

/*** Test5b ******************************************************************/

template <class T> struct Test5b {
  Test5b();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
template <> inline void Test5a<int>::bar();
template <> inline void Test5b<int>::bar() {}

// V-table should be defined with weak linkage.
template <> Test5b<int>::Test5b() { use(typeid(Test5b)); }
// CHECK: @_ZTV6Test5bIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test5bIiE = linkonce_odr constant
// CHECK: @_ZTI6Test5bIiE = linkonce_odr constant

template <> inline void Test5a<int>::foo();
template <> inline void Test5b<int>::foo() {}

/*** Test5c ******************************************************************/

template <class T> struct Test5c {
  Test5c();
  virtual void foo();
  virtual void bar();
};

// There ceases to be a key function after these declarations.
template <> inline void Test5a<int>::bar();
template <> inline void Test5a<int>::foo();
template <> inline void Test5c<int>::bar() {}
template <> inline void Test5c<int>::foo() {}

// V-table should be defined with weak linkage.
template <> Test5c<int>::Test5c() { use(typeid(Test5c)); }
// CHECK: @_ZTV6Test5cIiE = linkonce_odr unnamed_addr constant
// CHECK: @_ZTS6Test5cIiE = linkonce_odr constant
// CHECK: @_ZTI6Test5cIiE = linkonce_odr constant
