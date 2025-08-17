// Check that we can manually enable specific ptrauth features.

// RUN: %clang -target arm64-apple-darwin -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
// NONE: "-cc1"

// NONE-NOT: "-fptrauth-calls"
// NONE-NOT: "-fptrauth-returns"
// NONE-NOT: "-fptrauth-intrinsics"
// NONE-NOT: "-fptrauth-indirect-gotos"
// NONE-NOT: "-fptrauth-auth-traps"
// NONE-NOT: "-fptrauth-vtable-pointer-address-discrimination"
// NONE-NOT: "-fptrauth-vtable-pointer-type-discrimination"
// NONE-NOT: "-fptrauth-objc-isa"
// NONE-NOT: "-fptrauth-objc-class-ro"
// NONE-NOT: "-fptrauth-objc-interface-sel"

// Final catch all if any new flags are added
// NONE-NOT: "-fptrauth"

// RUN: %clang -target arm64-apple-darwin -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix CALL
// CALL: "-cc1"{{.*}} {{.*}} "-fptrauth-calls"

// RUN: %clang -target arm64-apple-darwin -fptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix INTRIN
// INTRIN: "-cc1"{{.*}} {{.*}} "-fptrauth-intrinsics"

// RUN: %clang -target arm64-apple-darwin -fptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix RETURN
// RETURN: "-cc1"{{.*}} {{.*}} "-fptrauth-returns"

// RUN: %clang -target arm64-apple-darwin -fptrauth-indirect-gotos -c %s -### 2>&1 | FileCheck %s --check-prefix INDGOTO
// INDGOTO: "-cc1"{{.*}} {{.*}} "-fptrauth-indirect-gotos"

// RUN: %clang -target arm64-apple-darwin -fptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix TRAPS
// TRAPS: "-cc1"{{.*}} {{.*}} "-fptrauth-auth-traps"


// Check the arm64e defaults.

// RUN: %clang -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -target arm64e-apple-macos -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULTMAC
// RUN: %clang -mkernel -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// RUN: %clang -fapple-kext -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// DEFAULT: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"{{.*}}
// DEFAULTMAC: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-m1"{{.*}}

// RUN: %clang -target arm64e-apple-none-macho -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-MACHO
// DEFAULT-MACHO: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"{{.*}}


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// DEFAULT-NOCALL-NOT: "-fptrauth-calls"
// DEFAULT-NOCALL: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix NORET

// NORET-NOT: "-fptrauth-returns"
// NORET: "-fptrauth-calls" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"

// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix NOINTRIN

// NOINTRIN-NOT: "-fptrauth-intrinsics"
// NOINTRIN: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// RUN: %clang -target arm64e-apple-ios -fno-ptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix NOTRAP
// NOTRAP: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel" {{.*}}"-target-cpu" "apple-a12"


// Check the CPU defaults and overrides.

// RUN: %clang -target arm64e-apple-ios -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A12
// RUN: %clang -target arm64e-apple-ios -mcpu=apple-a13 -c %s -### 2>&1 | FileCheck %s --check-prefix APPLE-A13
// APPLE-A12: "-cc1"{{.*}} "-target-cpu" "apple-a12"
// APPLE-A13: "-cc1"{{.*}} "-target-cpu" "apple-a13"

// Check that the default driver flags correctly propogate through to the compiler

// RUN: %clang -target arm64-apple-macosx      -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64-apple-ios         -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64e-apple-macosx     -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults
// RUN: %clang -target arm64e-apple-ios        -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults

// A simple test case to test basic override logic
// RUN: %clang -target arm64e-apple-macosx     -DDARWIN_DEFAULT_PTRAUTH_OVERRIDE -fno-ptrauth-calls %s -fsyntax-only -Xclang -verify=darwin_ptrauth_override


#define ASSERT_MODE_AND_KIND(feature, enabled, kind)                           \
  _Static_assert(enabled == __has_##kind(feature),                             \
                "Expected to have the " #feature " " #kind " enabled");

#define ASSERT_FEATURE_ENABLED(feature_name)                                   \
  ASSERT_MODE_AND_KIND(feature_name, 1, feature)
#define ASSERT_FEATURE_DISABLED(feature_name)                                  \
  ASSERT_MODE_AND_KIND(feature_name, 0, feature)
#define ASSERT_EXTENSION_ENABLED(extension_name)                               \
  ASSERT_MODE_AND_KIND(extension_name, 1, extension)
#define ASSERT_EXTENSION_DISABLED(extension_name)                              \
  ASSERT_MODE_AND_KIND(extension_name, 0, extension)

#if defined(DARWIN_DEFAULT_PTRAUTH) || defined(DARWIN_DEFAULT_PTRAUTH_OVERRIDE)
ASSERT_FEATURE_ENABLED(ptrauth_intrinsics)
ASSERT_EXTENSION_ENABLED(ptrauth_qualifier)

#if defined(DARWIN_DEFAULT_PTRAUTH_OVERRIDE)
ASSERT_FEATURE_DISABLED(ptrauth_calls)
// These flags directly reflect the state of ptrauth_calls, but exist
// for backward compatibility reasons
ASSERT_FEATURE_DISABLED(ptrauth_member_function_pointer_type_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_objc_method_list_pointer)
#else
ASSERT_FEATURE_ENABLED(ptrauth_calls)
ASSERT_FEATURE_ENABLED(ptrauth_member_function_pointer_type_discrimination)
ASSERT_FEATURE_ENABLED(ptrauth_objc_method_list_pointer)
#endif

ASSERT_FEATURE_ENABLED(ptrauth_returns)
ASSERT_FEATURE_ENABLED(ptrauth_vtable_pointer_address_discrimination)
ASSERT_FEATURE_ENABLED(ptrauth_vtable_pointer_type_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_type_info_vtable_pointer_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_signed_block_descriptors)
ASSERT_FEATURE_DISABLED(ptrauth_function_pointer_type_discrimination)
ASSERT_FEATURE_ENABLED(ptrauth_indirect_gotos)
ASSERT_FEATURE_DISABLED(ptrauth_init_fini)
ASSERT_FEATURE_DISABLED(ptrauth_init_fini_address_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_elf_got)
ASSERT_FEATURE_ENABLED(ptrauth_objc_isa)
ASSERT_FEATURE_ENABLED(ptrauth_objc_interface_sel)
ASSERT_FEATURE_ENABLED(ptrauth_objc_signable_class)
#endif

#ifdef NO_DEFAULT_PTRAUTH
ASSERT_FEATURE_DISABLED(ptrauth_intrinsics)
ASSERT_EXTENSION_DISABLED(ptrauth_qualifier)
ASSERT_FEATURE_DISABLED(ptrauth_calls)
ASSERT_FEATURE_DISABLED(ptrauth_returns)
ASSERT_FEATURE_DISABLED(ptrauth_vtable_pointer_address_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_vtable_pointer_type_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_type_info_vtable_pointer_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_member_function_pointer_type_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_signed_block_descriptors)
ASSERT_FEATURE_DISABLED(ptrauth_function_pointer_type_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_indirect_gotos)
ASSERT_FEATURE_DISABLED(ptrauth_init_fini)
ASSERT_FEATURE_DISABLED(ptrauth_init_fini_address_discrimination)
ASSERT_FEATURE_DISABLED(ptrauth_elf_got)
ASSERT_FEATURE_DISABLED(ptrauth_objc_isa)
ASSERT_FEATURE_DISABLED(ptrauth_objc_interface_sel)
ASSERT_FEATURE_DISABLED(ptrauth_objc_signable_class)
ASSERT_FEATURE_DISABLED(ptrauth_objc_method_list_pointer)
#endif

// darwin_ptrauth_defaults-no-diagnostics
// no_default_ptrauth-no-diagnostics
// darwin_ptrauth_override-no-diagnostics
