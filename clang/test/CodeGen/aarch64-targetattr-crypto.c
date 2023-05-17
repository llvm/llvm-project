// RUN: %clang_cc1 -triple aarch64-eabi -target-feature +v8a -verify -S %s -o -
// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

__attribute__((target("+crypto")))
void test_crypto(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

__attribute__((target("crypto")))
void test_pluscrypto(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

__attribute__((target("arch=armv8.2-a+crypto")))
void test_archcrypto(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

// FIXME: This shouldn't need +crypto to be consistent with -mcpu options.
__attribute__((target("cpu=cortex-a55+crypto")))
void test_a55crypto(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

__attribute__((target("cpu=cortex-a510+crypto")))
void test_a510crypto(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

__attribute__((target("+sha2+aes")))
void test_sha2aes(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key);
  vsha1su1q_u32(data, key);
}

void test_errors(uint8x16_t data, uint8x16_t key)
{
  vaeseq_u8(data, key); // expected-error {{always_inline function 'vaeseq_u8' requires target feature 'aes'}}
  vsha1su1q_u32(data, key); // expected-error {{always_inline function 'vsha1su1q_u32' requires target feature 'sha2'}}
}
