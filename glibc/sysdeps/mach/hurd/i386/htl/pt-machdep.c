/* Machine dependent pthreads code.  Hurd/i386 version.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library;  if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>

#include <mach.h>
#include <mach/i386/thread_status.h>
#include <mach/i386/mach_i386.h>
#include <mach/mig_errors.h>
#include <mach/thread_status.h>

#define HURD_TLS_DESC_DECL(desc, tcb)					      \
  struct descriptor desc =						      \
    {				/* low word: */				      \
      0xffff			/* limit 0..15 */			      \
      | (((unsigned int) (tcb)) << 16) /* base 0..15 */			      \
      ,				/* high word: */			      \
      ((((unsigned int) (tcb)) >> 16) & 0xff) /* base 16..23 */		      \
      | ((0x12 | 0x60 | 0x80) << 8) /* access = ACC_DATA_W|ACC_PL_U|ACC_P */  \
      | (0xf << 16)		/* limit 16..19 */			      \
      | ((4 | 8) << 20)		/* granularity = SZ_32|SZ_G */		      \
      | (((unsigned int) (tcb)) & 0xff000000) /* base 24..31 */		      \
    }

int
__thread_set_pcsptp (thread_t thread,
		     int set_ip, void *ip,
		     int set_sp, void *sp,
		     int set_tp, void *tp)
{
  error_t err;
  struct i386_thread_state state;
  mach_msg_type_number_t state_count;

  state_count = i386_THREAD_STATE_COUNT;

  err = __thread_get_state (thread, i386_REGS_SEGS_STATE,
			    (thread_state_t) &state, &state_count);
  if (err)
    return err;

  if (set_sp)
    state.uesp = (unsigned int) sp;
  if (set_ip)
    state.eip = (unsigned int) ip;
  if (set_tp)
    {
      HURD_TLS_DESC_DECL (desc, tp);
      int sel;

    asm ("mov %%gs, %w0": "=q" (sel):"0" (0));
      if (__builtin_expect (sel, 0x48) & 4)	/* LDT selector */
	err = __i386_set_ldt (thread, sel, &desc, 1);
      else
	err = __i386_set_gdt (thread, &sel, desc);
      if (err)
	return err;
      state.gs = sel;
    }

  err = __thread_set_state (thread, i386_REGS_SEGS_STATE,
			    (thread_state_t) &state, i386_THREAD_STATE_COUNT);
  if (err)
    return err;

  return 0;
}
