#include <sparc-ifunc.h>

#define  __md5_process_block __md5_process_block_generic
extern void __md5_process_block_generic (const void *buffer, size_t len,
					 struct md5_ctx *ctx);

#include <crypt/md5-block.c>

#undef __md5_process_block

extern void __md5_process_block_crop (const void *buffer, size_t len,
				      struct md5_ctx *ctx);
static bool cpu_supports_md5(int hwcap)
{
  unsigned long cfr;

  if (!(hwcap & HWCAP_SPARC_CRYPTO))
    return false;

  __asm__ ("rd %%asr26, %0" : "=r" (cfr));
  if (cfr & (1 << 4))
    return true;

  return false;
}

extern void __md5_process_block (const void *buffer, size_t len,
				 struct md5_ctx *ctx);
sparc_libc_ifunc(__md5_process_block, cpu_supports_md5(hwcap) ? __md5_process_block_crop : __md5_process_block_generic);
