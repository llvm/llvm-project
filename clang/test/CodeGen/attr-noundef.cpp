// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -x c++ -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-INTEL
// RUN: %clang_cc1 -no-opaque-pointers -triple aarch64-gnu-linux -x c++ -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AARCH
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -x c++ -S -emit-llvm -fsanitize-memory-param-retval %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-INTEL
// RUN: %clang_cc1 -no-opaque-pointers -triple aarch64-gnu-linux -x c++ -S -emit-llvm -fsanitize-memory-param-retval %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-AARCH

// no-sanitize-memory-param-retval does NOT conflict with enable-noundef-analysis
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -x c++ -S -emit-llvm -fno-sanitize-memory-param-retval %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-INTEL
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-gnu-linux -x c++ -S -emit-llvm -fno-sanitize-memory-param-retval %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-INTEL

//************ Passing structs by value
// TODO: No structs may currently be marked noundef

namespace check_structs {
struct Trivial {
  int a;
};
Trivial ret_trivial() { return {}; }
void pass_trivial(Trivial e) {}
// CHECK-INTEL: [[DEFINE:define( dso_local)?]] i32 @{{.*}}ret_trivial
// CHECK-AARCH: [[DEFINE:define( dso_local)?]] i32 @{{.*}}ret_trivial
// CHECK-INTEL: [[DEFINE]] void @{{.*}}pass_trivial{{.*}}(i32 %
// CHECK-AARCH: [[DEFINE]] void @{{.*}}pass_trivial{{.*}}(i64 %

struct NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}
// CHECK: [[DEFINE]] void @{{.*}}ret_nocopy{{.*}}(%"struct.check_structs::NoCopy"* noalias sret({{[^)]+}}) align 4 %
// CHECK: [[DEFINE]] void @{{.*}}pass_nocopy{{.*}}(%"struct.check_structs::NoCopy"* noundef %

struct Huge {
  int a[1024];
};
Huge ret_huge() { return {}; }
void pass_huge(Huge h) {}
// CHECK: [[DEFINE]] void @{{.*}}ret_huge{{.*}}(%"struct.check_structs::Huge"* noalias sret({{[^)]+}}) align 4 %
// CHECK: [[DEFINE]] void @{{.*}}pass_huge{{.*}}(%"struct.check_structs::Huge"* noundef
} // namespace check_structs

//************ Passing unions by value
// No unions may be marked noundef

namespace check_unions {
union Trivial {
  int a;
};
Trivial ret_trivial() { return {}; }
void pass_trivial(Trivial e) {}
// CHECK-INTEL: [[DEFINE]] i32 @{{.*}}ret_trivial
// CHECK-AARCH: [[DEFINE]] i32 @{{.*}}ret_trivial
// CHECK-INTEL: [[DEFINE]] void @{{.*}}pass_trivial{{.*}}(i32 %
// CHECK-AARCH: [[DEFINE]] void @{{.*}}pass_trivial{{.*}}(i64 %

union NoCopy {
  int a;
  NoCopy(NoCopy &) = delete;
};
NoCopy ret_nocopy() { return {}; }
void pass_nocopy(NoCopy e) {}
// CHECK: [[DEFINE]] void @{{.*}}ret_nocopy{{.*}}(%"union.check_unions::NoCopy"* noalias sret({{[^)]+}}) align 4 %
// CHECK: [[DEFINE]] void @{{.*}}pass_nocopy{{.*}}(%"union.check_unions::NoCopy"* noundef %
} // namespace check_unions

//************ Passing `this` pointers
// `this` pointer must always be defined

namespace check_this {
struct Object {
  int data[];

  Object() {
    this->data[0] = 0;
  }
  int getData() {
    return this->data[0];
  }
  Object *getThis() {
    return this;
  }
};

void use_object() {
  Object obj;
  obj.getData();
  obj.getThis();
}
// CHECK: define linkonce_odr void @{{.*}}Object{{.*}}(%"struct.check_this::Object"* noundef nonnull align 4 dereferenceable(1) %
// CHECK: define linkonce_odr noundef i32 @{{.*}}Object{{.*}}getData{{.*}}(%"struct.check_this::Object"* noundef nonnull align 4 dereferenceable(1) %
// CHECK: define linkonce_odr noundef %"struct.check_this::Object"* @{{.*}}Object{{.*}}getThis{{.*}}(%"struct.check_this::Object"* noundef nonnull align 4 dereferenceable(1) %
} // namespace check_this

//************ Passing vector types

namespace check_vecs {
typedef int __attribute__((vector_size(12))) i32x3;
i32x3 ret_vec() {
  return {};
}
void pass_vec(i32x3 v) {
}

// CHECK: [[DEFINE]] noundef <3 x i32> @{{.*}}ret_vec{{.*}}()
// CHECK-INTEL: [[DEFINE]] void @{{.*}}pass_vec{{.*}}(<3 x i32> noundef %
// CHECK-AARCH: [[DEFINE]] void @{{.*}}pass_vec{{.*}}(<4 x i32> %
} // namespace check_vecs

//************ Passing exotic types
// Function/Array pointers, Function member / Data member pointers, nullptr_t, ExtInt types

namespace check_exotic {
struct Object {
  int mfunc();
  int mdata;
};
typedef int Object::*mdptr;
typedef int (Object::*mfptr)();
typedef decltype(nullptr) nullptr_t;
typedef int (*arrptr)[32];
typedef int (*fnptr)(int);

arrptr ret_arrptr() {
  return nullptr;
}
fnptr ret_fnptr() {
  return nullptr;
}
mdptr ret_mdptr() {
  return nullptr;
}
mfptr ret_mfptr() {
  return nullptr;
}
nullptr_t ret_npt() {
  return nullptr;
}
void pass_npt(nullptr_t t) {
}
_BitInt(3) ret_BitInt() {
  return 0;
}
void pass_BitInt(_BitInt(3) e) {
}
void pass_large_BitInt(_BitInt(127) e) {
}

// Pointers to arrays/functions are always noundef
// CHECK: [[DEFINE]] noundef [32 x i32]* @{{.*}}ret_arrptr{{.*}}()
// CHECK: [[DEFINE]] noundef i32 (i32)* @{{.*}}ret_fnptr{{.*}}()

// Pointers to members are never noundef
// CHECK: [[DEFINE]] i64 @{{.*}}ret_mdptr{{.*}}()
// CHECK-INTEL: [[DEFINE]] { i64, i64 } @{{.*}}ret_mfptr{{.*}}()
// CHECK-AARCH: [[DEFINE]] [2 x i64] @{{.*}}ret_mfptr{{.*}}()

// nullptr_t is never noundef
// CHECK: [[DEFINE]] i8* @{{.*}}ret_npt{{.*}}()
// CHECK: [[DEFINE]] void @{{.*}}pass_npt{{.*}}(i8* %

// TODO: for now, ExtInt is only noundef if it is sign/zero-extended
// CHECK-INTEL: [[DEFINE]] noundef signext i3 @{{.*}}ret_BitInt{{.*}}()
// CHECK-AARCH: [[DEFINE]] i3 @{{.*}}ret_BitInt{{.*}}()
// CHECK-INTEL: [[DEFINE]] void @{{.*}}pass_BitInt{{.*}}(i3 noundef signext %
// CHECK-AARCH: [[DEFINE]] void @{{.*}}pass_BitInt{{.*}}(i3 %
// CHECK-INTEL: [[DEFINE]] void @{{.*}}pass_large_BitInt{{.*}}(i64 %{{.*}}, i64 %
// CHECK-AARCH: [[DEFINE]] void @{{.*}}pass_large_BitInt{{.*}}(i127 %
} // namespace check_exotic
