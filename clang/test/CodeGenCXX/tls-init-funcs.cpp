// RUN: %clang_cc1 -triple x86_64-apple-macosx10.8 -std=c++1y -S -emit-llvm %s -o - | FileCheck %s

// CHECK: @a = internal thread_local global
// CHECK: @_Z2vtIiE = linkonce_odr thread_local global i32 5
// CHECK: @_ZZ3inlvE3loc = linkonce_odr thread_local global i32 0
// CHECK: @_tlv_atexit({{.*}}@_ZN1AD1Ev
// CHECK: call cxx_fast_tlscc i32* @_ZTW3ext()
// CHECK: declare cxx_fast_tlscc i32* @_ZTW3ext()
// CHECK: define weak_odr hidden cxx_fast_tlscc i32* @_ZTW2vtIiE()
// CHECK: define weak_odr hidden cxx_fast_tlscc i32* @_ZTW2vtIvE()
// CHECK: define {{.*}} @_ZTW1a

struct A {
  ~A();
};

thread_local A a;

extern thread_local int ext;
int &get_ext() { return ext; }

template <typename T>
thread_local int vt = 5;

int get_vt() { return vt<int>; }

inline int &inl() {
  thread_local int loc;
  return loc;
}
int &use_inl() { return inl(); }

template int vt<void>;
int &get_vt_void() { return vt<void>; }
