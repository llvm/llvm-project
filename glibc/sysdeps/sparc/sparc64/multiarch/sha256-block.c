#include <sparc-ifunc.h>

#define __sha256_process_block __sha256_process_block_generic
extern void __sha256_process_block_generic (const void *buffer, size_t len,
					    struct sha256_ctx *ctx);

#include <crypt/sha256-block.c>

#undef __sha256_process_block

extern void __sha256_process_block_crop (const void *buffer, size_t len,
					 struct sha256_ctx *ctx);

static bool cpu_supports_sha256(int hwcap)
{
  unsigned long cfr;

  if (!(hwcap & HWCAP_SPARC_CRYPTO))
    return false;

  __asm__ ("rd %%asr26, %0" : "=r" (cfr));
  if (cfr & (1 << 6))
    return true;

  return false;
}

extern void __sha256_process_block (const void *buffer, size_t len,
				    struct sha256_ctx *ctx);
sparc_libc_ifunc (__sha256_process_block,
		  cpu_supports_sha256(hwcap) ? __sha256_process_block_crop
		    : __sha256_process_block_generic);
