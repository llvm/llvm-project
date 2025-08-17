// RUN: %clang -target arm64                   -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64-apple-macosx      -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64-darwin            -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64-apple-darwin      -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64-apple-ios-macabi  -DNO_DEFAULT_PTRAUTH     %s -fsyntax-only -Xclang -verify=no_default_ptrauth
// RUN: %clang -target arm64e-apple-macosx     -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults
// RUN: %clang -target arm64e-apple-ios        -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults
// RUN: %clang -target arm64e-darwin           -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults
// RUN: %clang -target arm64e-apple-darwin     -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults
// RUN: %clang -target arm64e-apple-ios-macabi -DDARWIN_DEFAULT_PTRAUTH %s -fsyntax-only -Xclang -verify=darwin_ptrauth_defaults

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
