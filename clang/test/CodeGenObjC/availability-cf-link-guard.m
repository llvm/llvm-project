// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CHECK_LINK_OPT %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11 -emit-llvm -o - -D USE_BUILTIN %s | FileCheck --check-prefixes=CHECK,CHECK_LINK_OPT %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.11 -emit-llvm -o - -D DEF_CF %s | FileCheck --check-prefixes=CHECK_CF,CHECK_LINK_OPT %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.15 -DCHECK_OS="macos 10.15.1" -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple arm64-apple-ios13.0 -DCHECK_OS="ios 14" -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple arm64-apple-tvos13.0 -DCHECK_OS="tvos 14" -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple arm64-apple-watchos6.0 -DCHECK_OS="watchos 7" -emit-llvm -o - %s | FileCheck --check-prefix=CHECK_NO_GUARD %s
// RUN: %clang_cc1 -triple arm64-apple-ios12.0 -DCHECK_OS="ios 13" -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CHECK_LINK_OPT %s
// RUN: %clang_cc1 -triple arm64-apple-tvos12.0 -DCHECK_OS="tvos 13" -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CHECK_LINK_OPT %s
// RUN: %clang_cc1 -triple arm64-apple-watchos5.0 -DCHECK_OS="watchos 6" -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CHECK_LINK_OPT %s

#ifdef DEF_CF
struct CFBundle;
typedef struct CFBundle *CFBundleRef;
unsigned CFBundleGetVersionNumber(CFBundleRef bundle);
// CHECK_CF: declare i32 @CFBundleGetVersionNumber(%struct.CFBundle* noundef)
// CHECK_CF: @__clang_at_available_requires_core_foundation_framework
// CHECK_CF-NEXT: call {{.*}}@CFBundleGetVersionNumber
#endif

#ifndef CHECK_OS
#define CHECK_OS macos 10.12
#endif

void use_at_available(void) {
#ifdef DEF_CF
  CFBundleGetVersionNumber(0);
#endif
#ifdef USE_BUILTIN
  if (__builtin_available(CHECK_OS, *))
    ;
#else
  if (@available(CHECK_OS, *))
    ;
#endif
}

// CHECK: @llvm.compiler.used{{.*}}@__clang_at_available_requires_core_foundation_framework

// CHECK: declare i32 @CFBundleGetVersionNumber(i8*)

// CHECK-LABEL: linkonce hidden void @__clang_at_available_requires_core_foundation_framework
// CHECK: call i32 @CFBundleGetVersionNumber(i8* null)
// CHECK-NEXT: unreachable

// CHECK_NO_GUARD-NOT: __clang_at_available_requires_core_foundation_framework
// CHECK_NO_GUARD-NOT: CFBundleGetVersionNumber

// CHECK_LINK_OPT: !llvm.linker.options = !{![[FRAMEWORK:[0-9]+]]
// CHECK_LINK_OPT: ![[FRAMEWORK]] = !{!"-framework", !"CoreFoundation"}

// CHECK_NO_GUARD-NOT: !llvm.linker.options
// CHECK_NO_GUARD-NOT: CoreFoundation
