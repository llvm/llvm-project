// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -fsyntax-only -verify %s

// Check that we don't abort when SVE types are made nullable.  This
// interface is invalid anyway, but we won't diagnose that until the
// sizeless type extension is added.
@interface foo
@property(nullable) __SVInt8_t s8; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVInt16_t s16; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVInt32_t s32; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVInt64_t s64; // expected-error {{cannot be applied to non-pointer type}}

@property(nullable) __SVUint8_t u8; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVUint16_t u16; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVUint32_t u32; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVUint64_t u64; // expected-error {{cannot be applied to non-pointer type}}

@property(nullable) __SVFloat16_t f16; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVFloat32_t f32; // expected-error {{cannot be applied to non-pointer type}}
@property(nullable) __SVFloat64_t f64; // expected-error {{cannot be applied to non-pointer type}}

@property(nullable) __SVBfloat16_t bf16; // expected-error {{cannot be applied to non-pointer type}}

@property(nullable) __SVMfloat8_t mf8; // expected-error {{cannot be applied to non-pointer type}}

@property(nullable) __SVBool_t b8; // expected-error {{cannot be applied to non-pointer type}}
@end
