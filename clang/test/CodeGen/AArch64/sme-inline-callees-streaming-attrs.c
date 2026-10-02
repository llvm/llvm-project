// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -target-feature +sme -target-feature +sme2 %s -DUSE_FLATTEN -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -target-feature +sme -target-feature +sme2 %s -DUSE_ALWAYS_INLINE_STMT -o - | FileCheck %s

// RUN: %if cir-enabled %{%clang_cc1 -triple aarch64-none-linux-gnu -fclangir -emit-llvm -target-feature +sme -target-feature +sme2 %s -DUSE_ALWAYS_INLINE_STMT -o - | FileCheck %s %}
// RUN: %if cir-enabled %{%clang_cc1 -triple aarch64-none-linux-gnu -fclangir -emit-cir -target-feature +sme -target-feature +sme2 %s -DUSE_ALWAYS_INLINE_STMT -o - | FileCheck --check-prefixes=CIR %s %}

// TODO(cir): Add USE_FLATTEN run lines when CIR supports the `flatten` attribue.

// REQUIRES: aarch64-registered-target

extern void was_inlined(void);

#if defined(USE_FLATTEN)
    #define FN_ATTR __attribute__((flatten))
    #define STMT_ATTR
#elif defined(USE_ALWAYS_INLINE_STMT)
    #define FN_ATTR
    #define STMT_ATTR [[clang::always_inline]]
#else
    #error Expected USE_FLATTEN or USE_ALWAYS_INLINE_STMT to be defined.
#endif

void fn(void) { was_inlined(); }
void fn_streaming_compatible(void) __arm_streaming_compatible { was_inlined(); }
void fn_streaming(void) __arm_streaming { was_inlined(); }
__arm_locally_streaming void fn_locally_streaming(void) { was_inlined(); }
__arm_new("za") void fn_streaming_new_za(void) __arm_streaming { was_inlined(); }
__arm_new("zt0") void fn_streaming_new_zt0(void) __arm_streaming { was_inlined(); }

FN_ATTR
void caller(void) {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// CHECK-LABEL: void @caller()
//  CHECK:        call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming
//  CHECK-NEXT:   call void @fn_locally_streaming
//  CHECK-NEXT:   call void @fn_streaming_new_za
//  CHECK-NEXT:   call void @fn_streaming_new_zt0

// CIR-LABEL: @caller()
// CIR: cir.call @fn() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming_compatible() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_locally_streaming()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_za()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_zt0()
// CIR-NOT: inline_kind

FN_ATTR void caller_streaming_compatible(void) __arm_streaming_compatible {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// CHECK-LABEL: void @caller_streaming_compatible()
//  CHECK:        call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming
//  CHECK-NEXT:   call void @fn_locally_streaming
//  CHECK-NEXT:   call void @fn_streaming_new_za
//  CHECK-NEXT:   call void @fn_streaming_new_zt0

// CIR-LABEL: @caller_streaming_compatible()
// CIR: cir.call @fn()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_compatible() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_locally_streaming()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_za()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_zt0()
// CIR-NOT: inline_kind

FN_ATTR void caller_streaming(void) __arm_streaming {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// CHECK-LABEL: void @caller_streaming()
//  CHECK:        call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming_new_za
//  CHECK-NEXT:   call void @fn_streaming_new_zt0

// CIR-LABEL: @caller_streaming()
// CIR: cir.call @fn()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_compatible() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_locally_streaming() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming_new_za()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_zt0()
// CIR-NOT: inline_kind

FN_ATTR __arm_locally_streaming
void caller_locally_streaming(void) {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// CHECK-LABEL: void @caller_locally_streaming()
//  CHECK:        call void @fn
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @was_inlined
//  CHECK-NEXT:   call void @fn_streaming_new_za
//  CHECK-NEXT:   call void @fn_streaming_new_zt0

// CIR-LABEL: @caller_locally_streaming()
// CIR: cir.call @fn()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_compatible() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_locally_streaming() {inline_kind = #cir.inline<always_inline>}
// CIR: cir.call @fn_streaming_new_za()
// CIR-NOT: inline_kind
// CIR: cir.call @fn_streaming_new_zt0()
// CIR-NOT: inline_kind
