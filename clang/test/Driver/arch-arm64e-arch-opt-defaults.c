// REQUIRES: system-darwin && target={{.*}}-{{darwin|macos}}{{.*}}
// Check the correct ptrauth features are enabled when simply using the -arch
// option

// RUN: %clang -arch arm64 -c %s -### 2>&1 | FileCheck %s --check-prefix NONE
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

// RUN: %clang -arch arm64 -fptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix CALL
// CALL: "-cc1"{{.*}} {{.*}} "-fptrauth-calls"

// RUN: %clang -arch arm64 -fptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix INTRIN
// INTRIN: "-cc1"{{.*}} {{.*}} "-fptrauth-intrinsics"

// RUN: %clang -arch arm64 -fptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix RETURN
// RETURN: "-cc1"{{.*}} {{.*}} "-fptrauth-returns"

// RUN: %clang -arch arm64 -fptrauth-indirect-gotos -c %s -### 2>&1 | FileCheck %s --check-prefix INDGOTO
// INDGOTO: "-cc1"{{.*}} {{.*}} "-fptrauth-indirect-gotos"

// RUN: %clang -arch arm64 -fptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix TRAPS
// TRAPS: "-cc1"{{.*}} {{.*}} "-fptrauth-auth-traps"


// Check the `-arch arm64e` defaults.

// RUN: %clang -arch arm64e -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT
// DEFAULT: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"

// RUN: %clang -arch arm64e -fno-ptrauth-calls -c %s -### 2>&1 | FileCheck %s --check-prefix DEFAULT-NOCALL
// DEFAULT-NOCALL-NOT: "-fptrauth-calls"
// DEFAULT-NOCALL: "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"


// RUN: %clang -arch arm64e -fno-ptrauth-returns -c %s -### 2>&1 | FileCheck %s --check-prefix NORET

// NORET-NOT: "-fptrauth-returns"
// NORET: "-fptrauth-calls" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"

// RUN: %clang -arch arm64e -fno-ptrauth-intrinsics -c %s -### 2>&1 | FileCheck %s --check-prefix NOINTRIN

// NOINTRIN-NOT: "-fptrauth-intrinsics"
// NOINTRIN: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-indirect-gotos" "-fptrauth-auth-traps" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"


// RUN: %clang -arch arm64e -fno-ptrauth-auth-traps -c %s -### 2>&1 | FileCheck %s --check-prefix NOTRAP
// NOTRAP: "-fptrauth-calls" "-fptrauth-returns" "-fptrauth-intrinsics" "-fptrauth-indirect-gotos" "-fptrauth-vtable-pointer-address-discrimination" "-fptrauth-vtable-pointer-type-discrimination" "-fptrauth-objc-isa" "-fptrauth-objc-class-ro" "-fptrauth-objc-interface-sel"

// Check that the default driver flags correctly propogate through to the compiler

// RUN: %clang -arch arm64                   -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -arch arm64e                  -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults

// A simple test case to test basic override logic
// RUN: %clang -arch arm64e                  -DDARWIN_DEFAULT_PTRAUTH_OVERRIDE -fno-ptrauth-calls %s -fsyntax-only -Xclang -verify=darwin_ptrauth_override


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
