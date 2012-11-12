// For MSVC ABI compatibility, all structures returned by value using the
// thiscall calling convention must use the hidden parameter.
//
// RUN: %clang_cc1 -triple i386-PC-Win32 %s -fms-compatibility -O0 -emit-llvm -o - | FileCheck %s

// This structure would normally be returned via EAX
struct S {
  int i;
};

// This structure would normally be returned via EAX/EDX
struct M {
  int i;
  int j;
};

class C {
public:
  C() {}

  struct S __attribute__((thiscall)) Small() const {
    struct S s = { 0 };
    return s;
  }

  struct M __attribute__((thiscall)) Medium() const {
    struct M m = { 0 };
    return m;
  }
};

// CHECK: define void @_Z4testv()
void test( void ) {
// CHECK: call void @_ZN1CC1Ev(%class.C* [[C:%.+]])
  C c;

// CHECK: call x86_thiscallcc void @_ZNK1C5SmallEv(%struct.S* sret %{{.+}}, %class.C* [[C]])
  (void)c.Small();
// CHECK: call x86_thiscallcc void @_ZNK1C6MediumEv(%struct.M* sret %{{.+}}, %class.C* [[C]])
  (void)c.Medium();
}
