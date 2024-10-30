// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @used_before_default_def = weak_odr ifunc void (), ptr @used_before_default_def.resolver
// CHECK: @used_after_default_def = weak_odr ifunc void (), ptr @used_after_default_def.resolver
// CHECK-NOT: @used_before_default_decl = weak_odr ifunc void (), ptr @used_before_default_decl.resolver
// CHECK-NOT: @used_after_default_decl = weak_odr ifunc void (), ptr @used_after_default_decl.resolver
// CHECK-NOT: @used_no_default = weak_odr ifunc void (), ptr @used_no_default.resolver
// CHECK-NOT: @not_used_no_default = weak_odr ifunc void (), ptr @not_used_no_default.resolver
// CHECK: @not_used_with_default = weak_odr ifunc void (), ptr @not_used_with_default.resolver


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


// CHECK: define weak_odr ptr @used_before_default_def.resolver()
// CHECK: define weak_odr ptr @used_after_default_def.resolver()
// CHECK-NOT: define weak_odr ptr @used_before_default_decl.resolver(
// CHECK-NOT: define weak_odr ptr @used_after_default_decl.resolver(
// CHECK-NOT: define weak_odr ptr @used_no_default.resolver(
// CHECK-NOT: define weak_odr ptr @not_used_no_default.resolver(
// CHECK: define weak_odr ptr @not_used_with_default.resolver()
