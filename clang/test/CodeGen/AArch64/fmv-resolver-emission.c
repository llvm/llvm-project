// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @used_before_default_def = weak_odr ifunc void (), ptr @used_before_default_def.resolver
//
// CHECK: @used_after_default_def = weak_odr ifunc void (), ptr @used_after_default_def.resolver
//
// CHECK-NOT: @used_before_default_decl = weak_odr ifunc void (), ptr @used_before_default_decl.resolver
// CHECK-NOT: @used_after_default_decl = weak_odr ifunc void (), ptr @used_after_default_decl.resolver
// CHECK-NOT: @used_no_default = weak_odr ifunc void (), ptr @used_no_default.resolver
// CHECK-NOT: @not_used_no_default = weak_odr ifunc void (), ptr @not_used_no_default.resolver
//
// CHECK: @not_used_with_default = weak_odr ifunc void (), ptr @not_used_with_default.resolver
//
// CHECK: @indirect_use = weak_odr ifunc void (), ptr @indirect_use.resolver
//
// CHECK: @internal_func = internal ifunc void (), ptr @internal_func.resolver
//
// CHECK: @linkonce_func = weak_odr ifunc void (), ptr @linkonce_func.resolver


// Test that an ifunc is generated and used when the default
// version is defined after the first use of the function.
//
__attribute__((target_version("aes"))) void used_before_default_def(void) {}
// CHECK-LABEL: define dso_local void @used_before_default_def._Maes(
//
void call_before_def(void) { used_before_default_def(); }
// CHECK-LABEL: define dso_local void @call_before_def(
// CHECK: call void @used_before_default_def()
//
__attribute__((target_version("default"))) void used_before_default_def(void) {}
// CHECK-LABEL: define dso_local void @used_before_default_def.default(
//
// CHECK-NOT: declare void @used_before_default_def(


// Test that an ifunc is generated and used when the default
// version is defined before the first use of the function.
//
__attribute__((target_version("aes"))) void used_after_default_def(void) {}
// CHECK-LABEL: define dso_local void @used_after_default_def._Maes(
//
__attribute__((target_version("default"))) void used_after_default_def(void) {}
// CHECK-LABEL: define dso_local void @used_after_default_def.default(
//
void call_after_def(void) { used_after_default_def(); }
// CHECK-LABEL: define dso_local void @call_after_def(
// CHECK: call void @used_after_default_def()
//
// CHECK-NOT: declare void @used_after_default_def(


// Test that an unmagled declaration is generated and used when the
// default version is declared after the first use of the function.
//
__attribute__((target_version("aes"))) void used_before_default_decl(void) {}
// CHECK-LABEL: define dso_local void @used_before_default_decl._Maes(
//
void call_before_decl(void) { used_before_default_decl(); }
// CHECK-LABEL: define dso_local void @call_before_decl(
// CHECK: call void @used_before_default_decl()
//
__attribute__((target_version("default"))) void used_before_default_decl(void);
// CHECK: declare void @used_before_default_decl()


// Test that an unmagled declaration is generated and used when the
// default version is declared before the first use of the function.
//
__attribute__((target_version("aes"))) void used_after_default_decl(void) {}
// CHECK-LABEL: define dso_local void @used_after_default_decl._Maes(
//
__attribute__((target_version("default"))) void used_after_default_decl(void);
// CHECK: declare void @used_after_default_decl()
//
void call_after_decl(void) { used_after_default_decl(); }
// CHECK-LABEL: define dso_local void @call_after_decl(
// CHECK: call void @used_after_default_decl()


// Test that an unmagled declaration is generated and used when
// the default version is not present.
//
__attribute__((target_version("aes"))) void used_no_default(void) {}
// CHECK-LABEL: define dso_local void @used_no_default._Maes(
//
void call_no_default(void) { used_no_default(); }
// CHECK-LABEL: define dso_local void @call_no_default(
// CHECK: call void @used_no_default()
//
// CHECK: declare void @used_no_default()


// Test that neither an ifunc nor a declaration is generated if the default
// definition is missing since the versioned function is not used.
//
__attribute__((target_version("aes"))) void not_used_no_default(void) {}
// CHECK-LABEL: define dso_local void @not_used_no_default._Maes(
//
// CHECK-NOT: declare void @not_used_no_default(


// Test that an ifunc is generated if the default version is defined but not used.
//
__attribute__((target_version("aes"))) void not_used_with_default(void) {}
// CHECK-LABEL: define dso_local void @not_used_with_default._Maes(
//
__attribute__((target_version("default"))) void not_used_with_default(void) {}
// CHECK-LABEL: define dso_local void @not_used_with_default.default(
//
// CHECK-NOT: declare void @not_used_with_default(


// Test that the ifunc symbol can be used for indirect calls.
//
__attribute__((target_version("aes"))) void indirect_use(void) {}
// CHECK-LABEL: define dso_local void @indirect_use._Maes(
//
__attribute__((target_version("default"))) void indirect_use(void) {}
// CHECK-LABEL: define dso_local void @indirect_use.default(
//
typedef void (*fptr)(void);
void call_indirectly(void) {
  fptr fn = indirect_use;
  fn();
}
// CHECK-LABEL: define dso_local void @call_indirectly(
// CHECK: [[FN:%.*]] = alloca ptr, align 8
// CHECK-NEXT: store ptr @indirect_use, ptr [[FN]], align 8
// CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[FN]], align 8
// CHECK-NEXT: call void [[TMP]]


// Test that an internal ifunc is generated if the versions are annotated with static inline.
//
static inline __attribute__((target_version("aes"))) void internal_func(void) {}
//
static inline __attribute__((target_version("default"))) void internal_func(void) {}
//
void call_internal(void) { internal_func(); }
// CHECK-LABEL: define dso_local void @call_internal(
// CHECK: call void @internal_func(


// Test that an ifunc is generated with if the versions are annotated with inline.
//
inline __attribute__((target_version("aes"))) void linkonce_func(void) {}
//
inline __attribute__((target_version("default"))) void linkonce_func(void) {}
//
void call_linkonce(void) { linkonce_func(); }
// CHECK-LABEL: define dso_local void @call_linkonce(
// CHECK: call void @linkonce_func(


// CHECK: define weak_odr ptr @used_before_default_def.resolver()
//
// CHECK: define weak_odr ptr @used_after_default_def.resolver()
//
// CHECK-NOT: define weak_odr ptr @used_before_default_decl.resolver(
// CHECK-NOT: define weak_odr ptr @used_after_default_decl.resolver(
// CHECK-NOT: define weak_odr ptr @used_no_default.resolver(
// CHECK-NOT: define weak_odr ptr @not_used_no_default.resolver(
//
// CHECK: define weak_odr ptr @not_used_with_default.resolver()
//
// CHECK: define weak_odr ptr @indirect_use.resolver()
//
// CHECK: define internal void @internal_func._Maes()
// CHECK: define internal void @internal_func.default()
// CHECK: define internal ptr @internal_func.resolver()
//
// CHECK: define linkonce void @linkonce_func._Maes()
// CHECK: define linkonce void @linkonce_func.default()
// CHECK: define weak_odr ptr @linkonce_func.resolver()
