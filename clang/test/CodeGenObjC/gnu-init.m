// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fno-use-init-array -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s -check-prefix=CHECK-NEW
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -S -emit-llvm -fno-use-init-array -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s -check-prefix=CHECK-WIN
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fno-use-init-array -fobjc-runtime=gnustep-1.8 -o - %s | FileCheck %s -check-prefix=CHECK-OLD
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s -check-prefix=CHECK-INIT_ARRAY

// Almost minimal Objective-C file, check that it emits calls to the correct
// runtime entry points.
@interface X @end
@implementation X @end


// Check that we emit a class ref
// CHECK-NEW: @._OBJC_REF_CLASS_X 
// CHECK-NEW-SAME: section "__objc_class_refs"

// Check that we get a class ref to the defined class.
// CHECK-NEW: @._OBJC_INIT_CLASS_X = global 
// CHECK-NEW-SAME: @._OBJC_CLASS_X, section "__objc_classes"

// Check that we emit the section start and end symbols as hidden globals.
// CHECK-NEW: @__start___objc_selectors = external hidden global ptr
// CHECK-NEW: @__stop___objc_selectors = external hidden global ptr
// CHECK-NEW: @__start___objc_classes = external hidden global ptr
// CHECK-NEW: @__stop___objc_classes = external hidden global ptr
// CHECK-NEW: @__start___objc_class_refs = external hidden global ptr
// CHECK-NEW: @__stop___objc_class_refs = external hidden global ptr
// CHECK-NEW: @__start___objc_cats = external hidden global ptr
// CHECK-NEW: @__stop___objc_cats = external hidden global ptr
// CHECK-NEW: @__start___objc_protocols = external hidden global ptr
// CHECK-NEW: @__stop___objc_protocols = external hidden global ptr
// CHECK-NEW: @__start___objc_protocol_refs = external hidden global ptr
// CHECK-NEW: @__stop___objc_protocol_refs = external hidden global ptr
// CHECK-NEW: @__start___objc_class_aliases = external hidden global ptr
// CHECK-NEW: @__stop___objc_class_aliases = external hidden global ptr
// CHECK-NEW: @__start___objc_constant_string = external hidden global ptr
// CHECK-NEW: @__stop___objc_constant_string = external hidden global ptr

// Check that we emit the init structure correctly, including in a comdat.
// CHECK-NEW: @.objc_init = linkonce_odr hidden global { i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i64 0, ptr @__start___objc_selectors, ptr @__stop___objc_selectors, ptr @__start___objc_classes, ptr @__stop___objc_classes, ptr @__start___objc_class_refs, ptr @__stop___objc_class_refs, ptr @__start___objc_cats, ptr @__stop___objc_cats, ptr @__start___objc_protocols, ptr @__stop___objc_protocols, ptr @__start___objc_protocol_refs, ptr @__stop___objc_protocol_refs, ptr @__start___objc_class_aliases, ptr @__stop___objc_class_aliases, ptr @__start___objc_constant_string, ptr @__stop___objc_constant_string }, comdat, align 8

// Check that the load function is manually inserted into .ctors.
// CHECK-NEW: @.objc_ctor = linkonce hidden global ptr @.objcv2_load_function, section ".ctors", comdat
// CHECK-INIT_ARRAY: @.objc_ctor = linkonce hidden global ptr @.objcv2_load_function, section ".init_array", comdat


// Make sure that we provide null versions of everything so the __start /
// __stop symbols work.
// CHECK-NEW: @.objc_null_selector = linkonce_odr hidden global { ptr, ptr } zeroinitializer, section "__objc_selectors", comdat, align 8
// CHECK-NEW: @.objc_null_category = linkonce_odr hidden global { ptr, ptr, ptr, ptr, ptr, ptr, ptr } zeroinitializer, section "__objc_cats", comdat, align 8
// CHECK-NEW: @.objc_null_protocol = linkonce_odr hidden global { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } zeroinitializer, section "__objc_protocols", comdat, align 8
// CHECK-NEW: @.objc_null_protocol_ref = linkonce_odr hidden global { ptr } zeroinitializer, section "__objc_protocol_refs", comdat, align 8
// CHECK-NEW: @.objc_null_class_alias = linkonce_odr hidden global { ptr, ptr } zeroinitializer, section "__objc_class_aliases", comdat, align 8
// CHECK-NEW: @.objc_null_constant_string = linkonce_odr hidden global { ptr, i32, i32, i32, i32, ptr } zeroinitializer, section "__objc_constant_string", comdat, align 8
// Make sure that the null symbols are not going to be removed, even by linking.
// CHECK-NEW: @llvm.used = appending global [8 x ptr] [ptr @._OBJC_INIT_CLASS_X, ptr @.objc_ctor, ptr @.objc_null_selector, ptr @.objc_null_category, ptr @.objc_null_protocol, ptr @.objc_null_protocol_ref, ptr @.objc_null_class_alias, ptr @.objc_null_constant_string], section "llvm.metadata"
// Make sure that the load function and the reference to it are marked as used.
// CHECK-NEW: @llvm.compiler.used = appending global [1 x ptr] [ptr @.objcv2_load_function], section "llvm.metadata"

// Check that we emit the load function in a comdat and that it does the right thing.
// CHECK-NEW: define linkonce_odr hidden void @.objcv2_load_function() comdat {
// CHECK-NEW-NEXT: entry:
// CHECK-NEW-NEXT: call void @__objc_load(
// CHECK-NEW-SAME: @.objc_init
// CHECK-NEW-NEXT: ret void

// CHECK-OLD: @4 = internal global { i64, i64, ptr, ptr } { i64 9, i64 32, ptr @.objc_source_file_name, ptr @3 }, align 8
// CHECK-OLD: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @.objc_load_function, ptr null }]

// CHECK-OLD: define internal void @.objc_load_function() {
// CHECK-OLD-NEXT: entry:
// CHECK-OLD-NEXT: call void (ptr, ...) @__objc_exec_class(ptr @4)



// Make sure all of our section boundary variables are emitted correctly.
// CHECK-WIN-DAG: @"__stop.objcrt$SEL" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$SEL$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$CLS" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CLS$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$CLS" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CLS$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$CLR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CLR$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$CLR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CLR$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$CAT" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CAT$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$CAT" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CAT$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$PCL" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$PCL$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$PCL" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$PCL$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$PCR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$PCR$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$PCR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$PCR$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$CAL" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CAL$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$CAL" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$CAL$z", comdat, align 8
// CHECK-WIN-DAG: @"__start_.objcrt$STR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$STR$a", comdat, align 8
// CHECK-WIN-DAG: @"__stop.objcrt$STR" = linkonce_odr hidden global %.objc_section_sentinel zeroinitializer, section ".objcrt$STR$z", comdat, align 8
// CHECK-WIN-DAG: @.objc_init = linkonce_odr hidden global { i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } { i64 0, ptr @"__start_.objcrt$SEL", ptr @"__stop.objcrt$SEL", ptr @"__start_.objcrt$CLS", ptr @"__stop.objcrt$CLS", ptr @"__start_.objcrt$CLR", ptr @"__stop.objcrt$CLR", ptr @"__start_.objcrt$CAT", ptr @"__stop.objcrt$CAT", ptr @"__start_.objcrt$PCL", ptr @"__stop.objcrt$PCL", ptr @"__start_.objcrt$PCR", ptr @"__stop.objcrt$PCR", ptr @"__start_.objcrt$CAL", ptr @"__stop.objcrt$CAL", ptr @"__start_.objcrt$STR", ptr @"__stop.objcrt$STR" }, comdat, align 8

// Make sure our init variable is in the correct section for late library init.
// CHECK-WIN: @.objc_ctor = linkonce hidden global ptr @.objcv2_load_function, section ".CRT$XCLz", comdat

// We shouldn't have emitted any null placeholders on Windows.
// CHECK-WIN: @llvm.used = appending global [2 x ptr] [ptr @"$_OBJC_INIT_CLASS_X", ptr @.objc_ctor], section "llvm.metadata"
// CHECK-WIN: @llvm.compiler.used = appending global [1 x ptr] [ptr @.objcv2_load_function], section "llvm.metadata"

// Check our load function is in a comdat.
// CHECK-WIN: define linkonce_odr hidden void @.objcv2_load_function() comdat {

// Make sure we do not have dllimport on the load function
// CHECK-WIN: declare dso_local void @__objc_load

