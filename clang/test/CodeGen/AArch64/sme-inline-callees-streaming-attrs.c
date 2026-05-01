// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -target-feature +sme -target-feature +sme2 %s -DUSE_FLATTEN -o - | FileCheck %s --check-prefix=CHECK-FLATTEN
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -target-feature +sme -target-feature +sme2 %s -DUSE_ALWAYS_INLINE_STMT -o - | FileCheck %s --check-prefix=CHECK-ALWAYS-INLINE

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
// For flatten: fn() and fn_streaming_compatible() are inlined, streaming functions
// are blocked by TTI (non-streaming caller), new_za/new_zt0 are always blocked.
// CHECK-FLATTEN-LABEL: void @caller()
//  CHECK-FLATTEN-NEXT: entry:
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming
//  CHECK-FLATTEN-NEXT:   call void @fn_locally_streaming
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_za
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_zt0

// For always_inline: Clang's wouldInliningViolateFunctionCallABI controls.
// CHECK-ALWAYS-INLINE-LABEL: void @caller()
//  CHECK-ALWAYS-INLINE-NEXT: entry:
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_locally_streaming
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_za
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_zt0

FN_ATTR void caller_streaming_compatible(void) __arm_streaming_compatible {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// For flatten: TTI allows inlining fn(), fn_streaming_compatible(), fn_streaming(),
// fn_locally_streaming() because they don't have incompatible ops. Only new_za/new_zt0 blocked.
// CHECK-FLATTEN-LABEL: void @caller_streaming_compatible()
//  CHECK-FLATTEN-NEXT: entry:
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_za
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_zt0

// For always_inline: Clang blocks fn() (streaming-compatible caller, non-streaming callee).
// CHECK-ALWAYS-INLINE-LABEL: void @caller_streaming_compatible()
//  CHECK-ALWAYS-INLINE-NEXT: entry:
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_locally_streaming
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_za
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_zt0

FN_ATTR void caller_streaming(void) __arm_streaming {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// For flatten: TTI allows all except new_za/new_zt0. fn() is inlined because
// streaming caller can execute non-streaming callee's code (no incompatible ops).
// CHECK-FLATTEN-LABEL: void @caller_streaming()
//  CHECK-FLATTEN-NEXT: entry:
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_za
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_zt0

// For always_inline: Clang blocks fn() (streaming caller, non-streaming callee).
// CHECK-ALWAYS-INLINE-LABEL: void @caller_streaming()
//  CHECK-ALWAYS-INLINE-NEXT: entry:
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_za
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_zt0

FN_ATTR __arm_locally_streaming
void caller_locally_streaming(void) {
    STMT_ATTR fn();
    STMT_ATTR fn_streaming_compatible();
    STMT_ATTR fn_streaming();
    STMT_ATTR fn_locally_streaming();
    STMT_ATTR fn_streaming_new_za();
    STMT_ATTR fn_streaming_new_zt0();
}
// For flatten: Similar to caller_streaming - TTI allows all except new_za/new_zt0.
// CHECK-FLATTEN-LABEL: void @caller_locally_streaming()
//  CHECK-FLATTEN-NEXT: entry:
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @was_inlined
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_za
//  CHECK-FLATTEN-NEXT:   call void @fn_streaming_new_zt0

// For always_inline: Clang blocks fn().
// CHECK-ALWAYS-INLINE-LABEL: void @caller_locally_streaming()
//  CHECK-ALWAYS-INLINE-NEXT: entry:
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @was_inlined
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_za
//  CHECK-ALWAYS-INLINE-NEXT:   call void @fn_streaming_new_zt0
