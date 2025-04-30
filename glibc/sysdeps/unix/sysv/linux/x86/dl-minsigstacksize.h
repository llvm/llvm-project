/* Emulate AT_MINSIGSTKSZ.  Linux/x86 version.
   Copyright (C) 2020 Free Software Foundation, Inc.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Emulate AT_MINSIGSTKSZ with XSAVE. */

static inline void
dl_check_minsigstacksize (const struct cpu_features *cpu_features)
{
  /* Return if AT_MINSIGSTKSZ is provide by kernel.  */
  if (GLRO(dl_minsigstacksize) != 0)
    return;

  if (cpu_features->basic.max_cpuid >= 0xd
      && CPU_FEATURES_CPU_P (cpu_features, OSXSAVE))
    {
      /* Emulate AT_MINSIGSTKSZ.  In Linux kernel, the signal frame data
	 with XSAVE is composed of the following areas and laid out as:
	 ------------------------------
	 | alignment padding          |
	 ------------------------------
	 | xsave buffer               |
	 ------------------------------
	 | fsave header (32-bit only) |
	 ------------------------------
	 | siginfo + ucontext         |
	 ------------------------------
	 */

      unsigned int sigframe_size;

#ifdef __x86_64__
      /* NB: sizeof(struct rt_sigframe) + 8-byte return address in Linux
	 kernel.  */
      sigframe_size = 440 + 8;
#else
      /* NB: sizeof(struct sigframe_ia32) + sizeof(struct fregs_state)) +
	 4-byte return address + 3 * 4-byte arguments in Linux kernel.  */
      sigframe_size = 736 + 112 + 4 + 3 * 4;
#endif

      /* Add 15 bytes to align the stack to 16 bytes.  */
      sigframe_size += 15;

      /* Make the space before xsave buffer multiple of 16 bytes.  */
      sigframe_size = ALIGN_UP (sigframe_size, 16);

      /* Add (64 - 16)-byte padding to align xsave buffer at 64 bytes.  */
      sigframe_size += 64 - 16;

      unsigned int eax, ebx, ecx, edx;
      __cpuid_count (0xd, 0, eax, ebx, ecx, edx);

      /* Add the size of xsave buffer.  */
      sigframe_size += ebx;

      /* Add the size of FP_XSTATE_MAGIC2.  */
#define FP_XSTATE_MAGIC2 0x46505845U
      sigframe_size += sizeof (FP_XSTATE_MAGIC2);

      GLRO(dl_minsigstacksize) = sigframe_size;
    }
  else
    {
      /* NB: Default to a constant MINSIGSTKSZ.  */
      _Static_assert (__builtin_constant_p (MINSIGSTKSZ),
		      "MINSIGSTKSZ is constant");
      GLRO(dl_minsigstacksize) = MINSIGSTKSZ;
    }
}
