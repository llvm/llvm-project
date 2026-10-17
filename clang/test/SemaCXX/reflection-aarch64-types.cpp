// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -std=c++26 -freflection -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -target-feature +sve -mvscale-min=1 -mvscale-max=1 -std=c++26 -freflection -fexperimental-new-constant-interpreter -fsyntax-only -verify %s
// expected-no-diagnostics

using Neon = signed char __attribute__((neon_vector_type(16)));
using NeonPoly = unsigned char __attribute__((neon_polyvector_type(16)));
static_assert(^^const Neon ==
              ^^const signed char __attribute__((neon_vector_type(16))));
static_assert(^^const Neon != ^^const NeonPoly);
static_assert(^^Neon * != ^^NeonPoly *);
static_assert(^^__SVInt8_t != ^^__SVUint8_t);
static_assert(^^__SVInt8_t != ^^__SVInt16_t);
static_assert(^^__SVInt8_t * != ^^__SVUint8_t *);
static_assert(^^__SVInt8_t != ^^const __SVInt8_t);

template <class T> consteval auto reflect() { return ^^T; }
static_assert(reflect<Neon>() ==
              ^^signed char __attribute__((neon_vector_type(16))));
static_assert(reflect<__SVInt8_t>() == ^^__typeof__(__SVInt8_t));
static_assert(reflect<__SVInt8_t>() != ^^__SVInt8_t);

using FixedSve = __SVInt8_t __attribute__((arm_sve_vector_bits(128)));
using OtherFixedSve = __SVUint8_t __attribute__((arm_sve_vector_bits(128)));
FixedSve fixed_value;
static_assert(^^FixedSve != ^^__typeof__(fixed_value));
static_assert(^^const FixedSve == ^^const __typeof__(fixed_value));
static_assert(^^FixedSve * != ^^OtherFixedSve *);
static_assert(^^FixedSve * != ^^__SVInt8_t *);
static_assert(reflect<FixedSve>() == ^^__typeof__(fixed_value));
