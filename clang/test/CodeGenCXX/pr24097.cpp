// RUN: %clang_cc1 %s -triple=x86_64-pc-linux -fvisibility hidden -emit-llvm -O1 -disable-llvm-passes -o - | FileCheck %s

struct Filter {
  virtual void Foo();
};
struct Sender {
  virtual bool Send();
};
struct SyncMessageFilter : public Filter, public Sender {
  bool Send();
};
struct TestSyncMessageFilter : public SyncMessageFilter {
};
void bar() {
  TestSyncMessageFilter f;
  f.Send();
}

// Test that it is not hidden
// CHECK: define available_externally noundef zeroext i1 @_ZThn8_N17SyncMessageFilter4SendEv
