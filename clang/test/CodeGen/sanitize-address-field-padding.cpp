// Test -fsanitize-address-field-padding
// RUN: echo 'type:SomeNamespace::IgnorelistedByName=field-padding' > %t.type.ignorelist
// RUN: echo 'src:*sanitize-address-field-padding.cpp=field-padding' > %t.file.ignorelist
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsanitize=address -fsanitize-address-field-padding=1 -fsanitize-ignorelist=%t.type.ignorelist -Rsanitize-address -emit-llvm -o - %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsanitize=address -fsanitize-address-field-padding=1 -fsanitize-ignorelist=%t.type.ignorelist -Rsanitize-address -emit-llvm -o - %s -mconstructor-aliases 2>&1 | FileCheck %s --check-prefix=WITH_CTOR_ALIASES
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsanitize=address -fsanitize-address-field-padding=1 -fsanitize-ignorelist=%t.file.ignorelist -Rsanitize-address -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=FILE_IGNORELIST
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s 2>&1 | FileCheck %s --check-prefix=NO_PADDING
// Try to emulate -save-temps option and make sure -disable-llvm-passes will not run sanitize instrumentation.
// RUN: %clang_cc1 -fsanitize=address -emit-llvm -disable-llvm-passes -o - %s | %clang_cc1 -fsanitize=address -emit-llvm -o - -x ir | FileCheck %s --check-prefix=NO_PADDING
//

// The reasons to ignore a particular class are not set in stone and will change.
//
// CHECK: -fsanitize-address-field-padding applied to Positive1
// CHECK: -fsanitize-address-field-padding ignored for Negative1 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative2 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative3 because it is a union
// CHECK: -fsanitize-address-field-padding ignored for Negative4 because it is trivially copyable
// CHECK: -fsanitize-address-field-padding ignored for Negative5 because it is packed
// CHECK: -fsanitize-address-field-padding ignored for SomeNamespace::IgnorelistedByName because it is ignorelisted
// CHECK: -fsanitize-address-field-padding ignored for ExternCStruct because it is not C++
//
// FILE_IGNORELIST: -fsanitize-address-field-padding ignored for Positive1 because it is in a ignorelisted file
// FILE_IGNORELIST-NOT: __asan_poison_intra_object_redzone
// NO_PADDING-NOT: __asan_poison_intra_object_redzone


class Positive1 {
 public:
  Positive1() {}
  ~Positive1() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
  short private_array[6];
  long long private3;
};

Positive1 positive1;
// Positive1 with extra paddings
// CHECK: type { i32, [12 x i8], i8, [15 x i8], i32, [12 x i8], [6 x i16], [12 x i8], i64, [8 x i8] }

struct VirtualBase {
  int foo;
};

class ClassWithVirtualBase : public virtual VirtualBase {
 public:
  ClassWithVirtualBase() {}
  ~ClassWithVirtualBase() {}
  int make_it_non_standard_layout;
 private:
  char x[7];
  char y[9];
};

ClassWithVirtualBase class_with_virtual_base;

class WithFlexibleArray1 {
 public:
  WithFlexibleArray1() {}
  ~WithFlexibleArray1() {}
  int make_it_non_standard_layout;
 private:
  char private1[33];
  int flexible[];  // Don't insert padding after this field.
};

WithFlexibleArray1 with_flexible_array1;
// CHECK: %class.WithFlexibleArray1 = type { i32, [12 x i8], [33 x i8], [15 x i8], [0 x i32] }

class WithFlexibleArray2 {
 public:
  char x[21];
  WithFlexibleArray1 flex1;  // Don't insert padding after this field.
};

WithFlexibleArray2 with_flexible_array2;
// CHECK: %class.WithFlexibleArray2 = type { [21 x i8], [11 x i8], %class.WithFlexibleArray1 }

class WithFlexibleArray3 {
 public:
  char x[13];
  WithFlexibleArray2 flex2;  // Don't insert padding after this field.
};

WithFlexibleArray3 with_flexible_array3;


class Negative1 {
 public:
  Negative1() {}
  int public1, public2;
};
Negative1 negative1;
// CHECK: type { i32, i32 }

class Negative2 {
 public:
  Negative2() {}
 private:
  int private1, private2;
};
Negative2 negative2;
// CHECK: type { i32, i32 }

union Negative3 {
  char m1[8];
  long long m2;
};

Negative3 negative3;
// CHECK: type { i64 }

class Negative4 {
 public:
  Negative4() {}
  // No DTOR
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};

Negative4 negative4;
// CHECK: type { i32, i8, i32 }

class __attribute__((packed)) Negative5 {
 public:
  Negative5() {}
  ~Negative5() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};

Negative5 negative5;
// CHECK: type <{ i32, i8, i32 }>


namespace SomeNamespace {
class IgnorelistedByName {
 public:
  IgnorelistedByName() {}
  ~IgnorelistedByName() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};
}  // SomeNamespace

SomeNamespace::IgnorelistedByName ignorelisted_by_name;

extern "C" {
class ExternCStruct {
 public:
  ExternCStruct() {}
  ~ExternCStruct() {}
  int make_it_non_standard_layout;
 private:
  char private1;
  int private2;
};
}  // extern "C"

ExternCStruct extern_C_struct;

// CTOR
// CHECK-LABEL: define {{.*}}Positive1C1Ev
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}15)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}}8)
// CHECK-NOT: __asan_poison_intra_object_redzone
// CHECK: ret void
//
// DTOR
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}15)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}12)
// CHECK: call void @__asan_unpoison_intra_object_redzone({{.*}}8)
// CHECK-NOT: __asan_unpoison_intra_object_redzone
// CHECK: ret void
//
//
// CHECK-LABEL: define linkonce_odr void @_ZN20ClassWithVirtualBaseC1Ev
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}} 12)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}} 9)
// CHECK: call void @__asan_poison_intra_object_redzone({{.*}} 15)
// CHECK-NOT: __asan_poison_intra_object_redzone
// CHECK: ret void
//

struct WithVirtualDtor {
  virtual ~WithVirtualDtor();
  int x, y;
};
struct InheritsFrom_WithVirtualDtor: WithVirtualDtor {
  int a, b;
  InheritsFrom_WithVirtualDtor() {}
  ~InheritsFrom_WithVirtualDtor() {}
};

void Create_InheritsFrom_WithVirtualDtor() {
  InheritsFrom_WithVirtualDtor x;
}


// Make sure the dtor of InheritsFrom_WithVirtualDtor remains in the code,
// i.e. we ignore -mconstructor-aliases when field paddings are added
// because the paddings in InheritsFrom_WithVirtualDtor needs to be unpoisoned
// in the dtor.
// WITH_CTOR_ALIASES-LABEL: define{{.*}} void @_Z35Create_InheritsFrom_WithVirtualDtor
// WITH_CTOR_ALIASES-NOT: call void @_ZN15WithVirtualDtorD2Ev
// WITH_CTOR_ALIASES: call void @_ZN28InheritsFrom_WithVirtualDtorD2Ev
// WITH_CTOR_ALIASES: ret void

// Make sure we don't emit memcpy for operator= if paddings are inserted.
struct ClassWithTrivialCopy {
  ClassWithTrivialCopy();
  ~ClassWithTrivialCopy();
  void *a;
 private:
  void *c;
};

void MakeTrivialCopy(ClassWithTrivialCopy *s1, ClassWithTrivialCopy *s2) {
  *s1 = *s2;
  ClassWithTrivialCopy s3(*s2);
}

// CHECK-LABEL: define{{.*}} void @_Z15MakeTrivialCopyP20ClassWithTrivialCopyS0_
// CHECK-NOT: memcpy
// CHECK: ret void
