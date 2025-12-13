// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=undefined %s -o - | FileCheck --check-prefix=CHECK-UBSAN %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=alignment %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-ALIGNMENT %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=bool %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-BOOL %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=builtin %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-BUILTIN %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=array-bounds %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-ARRAY-BOUNDS %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=enum %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-ENUM %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=float-cast-overflow %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-FLOAT-CAST-OVERFLOW %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=integer-divide-by-zero %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-INTEGER-DIVIDE-BY-ZERO %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=nonnull-attribute %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-NONNULL-ATTRIBUTE %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=null %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-NULL %s
// object-size is a no-op at O0.
// RUN: %clang -E -target x86_64-unknown-linux-gnu -O2 -fsanitize=object-size %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-OBJECT-SIZE %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=pointer-overflow %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-POINTER-OVERFLOW %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=return %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-RETURN %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=returns-nonnull-attribute %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-RETURNS-NONNULL-ATTRIBUTE %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=shift-base %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-SHIFT-BASE,CHECK-SHIFT %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=shift-exponent %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-SHIFT-EXPONENT,CHECK-SHIFT %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=shift %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-SHIFT %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-SIGNED-INTEGER-OVERFLOW %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=unreachable %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-UNREACHABLE %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=vla-bound %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-VLA-BOUND %s
// RUN: %clang -E -target x86_64-unknown-linux-gnu -fsanitize=function %s -o - | FileCheck --check-prefixes=CHECK-UBSAN,CHECK-FUNCTION %s

// RUN: %clang -E  %s -o - | FileCheck --check-prefix=CHECK-NO-UBSAN %s

// REQUIRES: x86-registered-target

#if !__has_feature(undefined_behavior_sanitizer_finegrained_feature_checks)
#error "Missing undefined_behavior_sanitizer_finegrained_feature_checks"
#endif

#if __has_feature(undefined_behavior_sanitizer)
int UBSanEnabled();
#else
int UBSanDisabled();
#endif

#if __has_feature(alignment_sanitizer)
int AlignmentSanitizerEnabled();
#else
int AlignmentSanitizerDisabled();
#endif

#if __has_feature(bool_sanitizer)
int BoolSanitizerEnabled();
#else
int BoolSanitizerDisabled();
#endif

#if __has_feature(builtin_sanitizer)
int BuiltinSanitizerEnabled();
#else
int BuiltinSanitizerDisabled();
#endif

#if __has_feature(array_bounds_sanitizer)
int ArrayBoundsSanitizerEnabled();
#else
int ArrayBoundsSanitizerDisabled();
#endif

#if __has_feature(enum_sanitizer)
int EnumSanitizerEnabled();
#else
int EnumSanitizerDisabled();
#endif

#if __has_feature(float_cast_overflow_sanitizer)
int FloatCastOverflowSanitizerEnabled();
#else
int FloatCastOverflowSanitizerDisabled();
#endif

#if __has_feature(integer_divide_by_zero_sanitizer)
int IntegerDivideByZeroSanitizerEnabled();
#else
int IntegerDivideByZeroSanitizerDisabled();
#endif

#if __has_feature(nonnull_attribute_sanitizer)
int NonnullAttributeSanitizerEnabled();
#else
int NonnullAttributeSanitizerDisabled();
#endif

#if __has_feature(null_sanitizer)
int NullSanitizerEnabled();
#else
int NullSanitizerDisabled();
#endif

#if __has_feature(object_size_sanitizer)
int ObjectSizeSanitizerEnabled();
#else
int ObjectSizeSanitizerDisabled();
#endif

#if __has_feature(pointer_overflow_sanitizer)
int PointerOverflowSanitizerEnabled();
#else
int PointerOverflowSanitizerDisabled();
#endif

#if __has_feature(return_sanitizer)
int ReturnSanitizerEnabled();
#else
int ReturnSanitizerDisabled();
#endif

#if __has_feature(returns_nonnull_attribute_sanitizer)
int ReturnsNonnullAttributeSanitizerEnabled();
#else
int ReturnsNonnullAttributeSanitizerDisabled();
#endif

#if __has_feature(shift_base_sanitizer)
int ShiftBaseSanitizerEnabled();
#else
int ShiftBaseSanitizerDisabled();
#endif

#if __has_feature(shift_exponent_sanitizer)
int ShiftExponentSanitizerEnabled();
#else
int ShiftExponentSanitizerDisabled();
#endif

#if __has_feature(shift_sanitizer)
int ShiftSanitizerEnabled();
#else
int ShiftSanitizerDisabled();
#endif

#if __has_feature(signed_integer_overflow_sanitizer)
int SignedIntegerOverflowSanitizerEnabled();
#else
int SignedIntegerOverflowSanitizerDisabled();
#endif

#if __has_feature(unreachable_sanitizer)
int UnreachableSanitizerEnabled();
#else
int UnreachableSanitizerDisabled();
#endif

#if __has_feature(vla_bound_sanitizer)
int VLABoundSanitizerEnabled();
#else
int VLABoundSanitizerDisabled();
#endif

#if __has_feature(function_sanitizer)
int FunctionSanitizerEnabled();
#else
int FunctionSanitizerDisabled();
#endif

// CHECK-UBSAN: UBSanEnabled
// CHECK-ALIGNMENT: AlignmentSanitizerEnabled
// CHECK-BOOL: BoolSanitizerEnabled
// CHECK-BUILTIN: BuiltinSanitizerEnabled
// CHECK-ARRAY-BOUNDS: ArrayBoundsSanitizerEnabled
// CHECK-ENUM: EnumSanitizerEnabled
// CHECK-FLOAT-CAST-OVERFLOW: FloatCastOverflowSanitizerEnabled
// CHECK-INTEGER-DIVIDE-BY-ZERO: IntegerDivideByZeroSanitizerEnabled
// CHECK-NONNULL-ATTRIBUTE: NonnullAttributeSanitizerEnabled
// CHECK-NULL: NullSanitizerEnabled
// CHECK-OBJECT-SIZE: ObjectSizeSanitizerEnabled
// CHECK-POINTER-OVERFLOW: PointerOverflowSanitizerEnabled
// CHECK-RETURN: ReturnSanitizerEnabled
// CHECK-RETURNS-NONNULL-ATTRIBUTE: ReturnsNonnullAttributeSanitizerEnabled
// CHECK-SHIFT-BASE: ShiftBaseSanitizerEnabled
// CHECK-SHIFT-EXPONENT: ShiftExponentSanitizerEnabled
// CHECK-SHIFT: ShiftSanitizerEnabled
// CHECK-SIGNED-INTEGER-OVERFLOW: SignedIntegerOverflowSanitizerEnabled
// CHECK-UNREACHABLE: UnreachableSanitizerEnabled
// CHECK-VLA-BOUND: VLABoundSanitizerEnabled
// CHECK-FUNCTION: FunctionSanitizerEnabled
// CHECK-NO-UBSAN: UBSanDisabled
