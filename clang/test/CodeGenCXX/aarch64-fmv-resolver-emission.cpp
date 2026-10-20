// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s | FileCheck %s

// CHECK: @_Z23used_before_default_defv = weak_odr ifunc void (), ptr @_Z23used_before_default_defv.resolver
// CHECK: @_Z22used_after_default_defv = weak_odr ifunc void (), ptr @_Z22used_after_default_defv.resolver
// CHECK-NOT: @_Z24used_before_default_declv = weak_odr ifunc void (), ptr @_Z24used_before_default_declv.resolver
// CHECK-NOT: @_Z23used_after_default_declv = weak_odr ifunc void (), ptr @_Z23used_after_default_declv.resolver
// CHECK-NOT: @_Z15used_no_defaultv = weak_odr ifunc void (), ptr @_Z15used_no_defaultv.resolver
// CHECK-NOT: @_Z19not_used_no_defaultv = weak_odr ifunc void (), ptr @_Z19not_used_no_defaultv.resolver
// CHECK: @_Z21not_used_with_defaultv = weak_odr ifunc void (), ptr @_Z21not_used_with_defaultv.resolver


// Test that an ifunc is generated and used when the default
// version is defined after the first use of the function.
//
__attribute__((target_version("aes"))) void used_before_default_def(void) {}
// CHECK-LABEL: define dso_local void @_Z23used_before_default_defv._Maes(
//
void call_before_def(void) { used_before_default_def(); }
// CHECK-LABEL: define dso_local void @_Z15call_before_defv(
// CHECK: call void @_Z23used_before_default_defv()
//
__attribute__((target_version("default"))) void used_before_default_def(void) {}
// CHECK-LABEL: define dso_local void @_Z23used_before_default_defv.default(
//
// CHECK-NOT: declare void @_Z23used_before_default_defv(


// Test that an ifunc is generated and used when the default
// version is defined before the first use of the function.
//
__attribute__((target_version("aes"))) void used_after_default_def(void) {}
// CHECK-LABEL: define dso_local void @_Z22used_after_default_defv._Maes(
//
__attribute__((target_version("default"))) void used_after_default_def(void) {}
// CHECK-LABEL: define dso_local void @_Z22used_after_default_defv.default(
//
void call_after_def(void) { used_after_default_def(); }
// CHECK-LABEL: define dso_local void @_Z14call_after_defv(
// CHECK: call void @_Z22used_after_default_defv()
//
// CHECK-NOT: declare void @_Z22used_after_default_defv(


// Test that an unmagled declaration is generated and used when the
// default version is declared after the first use of the function.
//
__attribute__((target_version("aes"))) void used_before_default_decl(void) {}
// CHECK-LABEL: define dso_local void @_Z24used_before_default_declv._Maes(
//
void call_before_decl(void) { used_before_default_decl(); }
// CHECK-LABEL: define dso_local void @_Z16call_before_declv(
// CHECK: call void @_Z24used_before_default_declv()
//
__attribute__((target_version("default"))) void used_before_default_decl(void);
// CHECK: declare void @_Z24used_before_default_declv()


// Test that an unmagled declaration is generated and used when the
// default version is declared before the first use of the function.
//
__attribute__((target_version("aes"))) void used_after_default_decl(void) {}
// CHECK-LABEL: define dso_local void @_Z23used_after_default_declv._Maes(
//
__attribute__((target_version("default"))) void used_after_default_decl(void);
// CHECK: declare void @_Z23used_after_default_declv()
//
void call_after_decl(void) { used_after_default_decl(); }
// CHECK-LABEL: define dso_local void @_Z15call_after_declv(
// CHECK: call void @_Z23used_after_default_declv()


// Test that an unmagled declaration is generated and used when
// the default version is not present.
//
__attribute__((target_version("aes"))) void used_no_default(void) {}
// CHECK-LABEL: define dso_local void @_Z15used_no_defaultv._Maes(
//
void call_no_default(void) { used_no_default(); }
// CHECK-LABEL: define dso_local void @_Z15call_no_defaultv(
// CHECK: call void @_Z15used_no_defaultv()
//
// CHECK: declare void @_Z15used_no_defaultv()


// Test that neither an ifunc nor a declaration is generated if the default
// definition is missing since the versioned function is not used.
//
__attribute__((target_version("aes"))) void not_used_no_default(void) {}
// CHECK-LABEL: define dso_local void @_Z19not_used_no_defaultv._Maes(
//
// CHECK-NOT: declare void @_Z19not_used_no_defaultv(


// Test that an ifunc is generated if the default version is defined but not used.
//
__attribute__((target_version("aes"))) void not_used_with_default(void) {}
// CHECK-LABEL: define dso_local void @_Z21not_used_with_defaultv._Maes(
//
__attribute__((target_version("default"))) void not_used_with_default(void) {}
// CHECK-LABEL: define dso_local void @_Z21not_used_with_defaultv.default(
//
// CHECK-NOT: declare void @_Z21not_used_with_defaultv(


// CHECK: define weak_odr ptr @_Z23used_before_default_defv.resolver()
// CHECK: define weak_odr ptr @_Z22used_after_default_defv.resolver()
// CHECK-NOT: define weak_odr ptr @_Z24used_before_default_declv.resolver(
// CHECK-NOT: define weak_odr ptr @_Z23used_after_default_declv.resolver(
// CHECK-NOT: define weak_odr ptr @_Z15used_no_defaultv.resolver(
// CHECK-NOT: define weak_odr ptr @_Z19not_used_no_defaultv.resolver(
// CHECK: define weak_odr ptr @_Z21not_used_with_defaultv.resolver()
