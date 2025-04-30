/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stdint.h>
#include "pgcpuid.h"

/*
 *      Note:
 *      1) these functions cannot call any other function
 *      2) these functions can only use GPR (not floating point)
 *
 */

/**     @brief returns false/true if CPUID supports eax function.
 *      __pgi_cpuid_getma (uint32_t cpuid_func)
 *      @param  cpuid_func (I1) function to execute CPUID with
 *
 *      Returns false(0)/true(1)
 *
 */

int
__pgi_cpuid_getmax(uint32_t f)
{
  static uint32_t maxcpueax[2] = { 0, 0 };
  uint32_t fin = f & 0x80000000;
  uint32_t findex = fin >> 31;	/* 0 or 1 */
  if (!maxcpueax[findex]) {
    asm("\tcpuid"
        : "=a"(maxcpueax[findex])
        : "0"(fin)
        : "ebx", "ecx", "edx"
        );
  }
  return f <= maxcpueax[findex];
}

/**     @brief returns results of executing CPUID with function cpuid_func and
 *      sub function ecx.
 *      __pgi_cpuid_ecx(uint32_t cpuid_func, uint32_t *res, uint32_t ecx)
 *      @param  cpuid_func (I1) function to execute CPUID with
 *      @param  res (I2) pointer to buffer to store eax, ebx, ecx, edx
 *      @param  ecx (I3) value of %ecx to execute CPUID with
 *
 *      Returns false(0): if cpuid_func not supported
 *              true(1):  CPUID successfully executed with cpuid_func+ecx and:
 *                        res[0]=%eax, res[1]=%ebx, res[2]=%ecx, res[3]=%edx
 *
 */

int
__pgi_cpuid_ecx(uint32_t f, uint32_t *r, uint32_t c)
{
  if (__pgi_cpuid_getmax(f) == 0) return 0;
  asm("\tcpuid"
        : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
        : "0"(f), "2"(c)
        );
  return 1;
}


/**     @brief returns results of executing CPUID with function cpuid_func.
 *      __pgi_cpuid(uint32_t cpuid_func, uint32_t *res)
 *      @param  cpuid_func (I1) function to execute CPUID with
 *      @param  res (I2) pointer to buffer to store eax, ebx, ecx, edx
 *
 *      Returns false(0): if cpuid_func not supported
 *              true(1):  CPUID successfully executed with cpuid_func and:
 *                        res[0]=%eax, res[1]=%ebx, res[2]=%ecx, res[3]=%edx
 *
 */

int
__pgi_cpuid(uint32_t f, uint32_t *r)
{
  return __pgi_cpuid_ecx(f, r, 0);
}

/**     @brief returns results of executing CPUID with function cpuid_func.
 *      __pgcpuid(uint32_t cpuid_func, uint32_t *res)
 *      @param  cpuid_func (I1) function to execute CPUID with
 *      @param  res (I2) pointer to buffer to store eax, ebx, ecx, edx
 *
 *      Returns false(0): if cpuid_func not supported
 *              true(1):  CPUID successfully executed with cpuid_func and:
 *                        res[0]=%eax, res[1]=%ebx, res[2]=%ecx, res[3]=%edx
 *
 */

int
__pgcpuid(uint32_t f, uint32_t *r)
{
  return __pgi_cpuid_ecx(f, r, 0);
}

/**     @brief read extended control register.
 *      __pgi_getbv(uint32_t xcr_num, uint64_t *xcr_res)
 *      @param  xcr_num (I1) extended control register number to read
 *      @param  xcr_res (I2) pointer to buffer to store xcr[xcr_num]
 *
 *      Returns true(1) with:
 *              xcr_res[31: 0]=%eax
 *              xcr_res[63:32]=%edx
 *
 */
int
__pgi_getbv(uint32_t f, uint64_t *r)
{
  uint32_t *u32;
  u32 = (uint32_t *)r;
  asm(
#if	defined(_WIN64)
"\t.byte\t0x0f, 0x01, 0xd0"
#else
"\txgetbv"
#endif
        : "=a"(u32[0]), "=d"(u32[1])
        : "c"(f)
        );
  return 1;
}
