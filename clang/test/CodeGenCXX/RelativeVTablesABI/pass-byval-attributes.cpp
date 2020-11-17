// ByVal attributes should propogate through to produce proper assembly and
// avoid "unpacking" structs within the stubs on x86_64.

// RUN: %clang_cc1 %s -triple=x86_64-unknown-fuchsia -S -o - -emit-llvm -fexperimental-relative-c++-abi-vtables | FileCheck %s

struct LargeStruct {
  char x[24];
  virtual ~LargeStruct() {}
};

struct fidl_string {
  unsigned long long size;
  char *data;
};
static_assert(sizeof(fidl_string) == 16, "");

class Base {
public:
  virtual void func(LargeStruct, fidl_string, LargeStruct, fidl_string) = 0;
};

class Derived : public Base {
public:
  void func(LargeStruct, fidl_string, LargeStruct, fidl_string) override;
};

// The original function takes a byval pointer.
// CHECK: define void @_ZN7Derived4funcE11LargeStruct11fidl_stringS0_S1_(%class.Derived* {{[^,]*}} %this, %struct.LargeStruct* %ls, i64 %sv1.coerce0, i8* %sv1.coerce1, %struct.LargeStruct* %ls2, %struct.fidl_string* byval(%struct.fidl_string) align 8 %sv2) unnamed_addr

// So the stub should take and pass one also.
// CHECK:      define hidden void @_ZN7Derived4funcE11LargeStruct11fidl_stringS0_S1_.stub(%class.Derived* {{[^,]*}} %0, %struct.LargeStruct* %1, i64 %2, i8* %3, %struct.LargeStruct* %4, %struct.fidl_string* byval(%struct.fidl_string) align 8 %5) unnamed_addr {{#[0-9]+}} comdat
// CHECK-NEXT: entry:
// CHECK-NEXT:   tail call void @_ZN7Derived4funcE11LargeStruct11fidl_stringS0_S1_(%class.Derived* {{[^,]*}} %0, %struct.LargeStruct* %1, i64 %2, i8* %3, %struct.LargeStruct* %4, %struct.fidl_string* byval(%struct.fidl_string) align 8 %5)
// CHECK-NEXT:   ret void
// CHECK-NEXT: }

void Derived::func(LargeStruct ls, fidl_string sv1, LargeStruct ls2, fidl_string sv2) {}
