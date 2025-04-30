/* Regularize <asm/unistd.h> definitions.  ARC version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.

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
   <http://www.gnu.org/licenses/>.  */

/* Adjustments to ARC asm-generic syscall ABI (3.9 kernel) for 64-bit time_t
   support.  */

/* fstat64 and fstatat64 need to be replaced with statx.  */

#undef __NR_fstat64
#undef __NR_fstatat64

/* Elide all other 32-bit time_t syscalls.  */

# undef __NR_clock_adjtime
# undef __NR_clock_getres
# undef __NR_clock_gettime
# undef __NR_clock_nanosleep
# undef __NR_clock_settime
# undef __NR_futex
# undef __NR_mq_timedreceive
# undef __NR_mq_timedsend
# undef __NR_ppoll
# undef __NR_pselect6
# undef __NR_recvmmsg
# undef __NR_rt_sigtimedwait
# undef __NR_sched_rr_get_interval
# undef __NR_semtimedop
# undef __NR_timer_gettime
# undef __NR_timer_settime
# undef __NR_timerfd_gettime
# undef __NR_timerfd_settime
# undef __NR_utimensat
