// The -cc1 mode should not insert default ptrauth flags

// RUN: %clang_cc1 -triple arm64                   %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64-apple-macosx      %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64-apple-ios         %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64-apple-ios-macabi  %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64e                  %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64e-apple-macosx     %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64e-apple-ios        %s -fsyntax-only
// RUN: %clang_cc1 -triple arm64e-apple-ios-macabi %s -fsyntax-only

#define ASSERT_MODE_AND_KIND(feature, enabled, kind)                           \
  _Static_assert(enabled == __has_##kind(feature),                             \
                "Expected to have the " #feature " " #kind " enabled");

#define ASSERT_FEATURE_DISABLED(feature_name)                                  \
  ASSERT_MODE_AND_KIND(feature_name, 0, feature)

ASSERT_FEATURE_DISABLED(ptrauth_intrinsics)
ASSERT_FEATURE_DISABLED(ptrauth_qualifier)
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

// expected-no-diagnostics
