/* Iterate over a process's threads.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 1999.

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

#include "thread_dbP.h"


static td_err_e
iterate_thread_list (td_thragent_t *ta, td_thr_iter_f *callback,
		     void *cbdata_p, td_thr_state_e state, int ti_pri,
		     psaddr_t head, bool fake_empty)
{
  td_err_e err;
  psaddr_t next, ofs;
  void *copy;

  /* Test the state.
     XXX This is incomplete.  Normally this test should be in the loop.  */
  if (state != TD_THR_ANY_STATE)
    return TD_OK;

  err = DB_GET_FIELD (next, ta, head, list_t, next, 0);
  if (err != TD_OK)
    return err;

  if (next == 0 && fake_empty)
    {
      /* __pthread_initialize_minimal has not run.  There is just the main
	 thread to return.  We cannot rely on its thread register.  They
	 sometimes contain garbage that would confuse us, left by the
	 kernel at exec.  So if it looks like initialization is incomplete,
	 we only fake a special descriptor for the initial thread.  */
      td_thrhandle_t th = { ta, 0 };
      return callback (&th, cbdata_p) != 0 ? TD_DBERR : TD_OK;
    }

  /* Cache the offset from struct pthread to its list_t member.  */
  err = DB_GET_FIELD_ADDRESS (ofs, ta, 0, pthread, list, 0);
  if (err != TD_OK)
    return err;

  if (ta->ta_sizeof_pthread == 0)
    {
      err = _td_check_sizeof (ta, &ta->ta_sizeof_pthread, SYM_SIZEOF_pthread);
      if (err != TD_OK)
	return err;
    }
  copy = __alloca (ta->ta_sizeof_pthread);

  while (next != head)
    {
      psaddr_t addr, schedpolicy, schedprio;

      addr = next - (ofs - (psaddr_t) 0);
      if (next == 0 || addr == 0) /* Sanity check.  */
	return TD_DBERR;

      /* Copy the whole descriptor in once so we can access the several
	 fields locally.  Excess copying in one go is much better than
	 multiple ps_pdread calls.  */
      if (ps_pdread (ta->ph, addr, copy, ta->ta_sizeof_pthread) != PS_OK)
	return TD_ERR;

      err = DB_GET_FIELD_LOCAL (schedpolicy, ta, copy, pthread,
				schedpolicy, 0);
      if (err != TD_OK)
	break;
      err = DB_GET_FIELD_LOCAL (schedprio, ta, copy, pthread,
				schedparam_sched_priority, 0);
      if (err != TD_OK)
	break;

      /* Now test whether this thread matches the specified conditions.  */

      /* Only if the priority level is as high or higher.  */
      int descr_pri = ((uintptr_t) schedpolicy == SCHED_OTHER
		       ? 0 : (uintptr_t) schedprio);
      if (descr_pri >= ti_pri)
	{
	  /* Yep, it matches.  Call the callback function.  */
	  td_thrhandle_t th;
	  th.th_ta_p = (td_thragent_t *) ta;
	  th.th_unique = addr;
	  if (callback (&th, cbdata_p) != 0)
	    return TD_DBERR;
	}

      /* Get the pointer to the next element.  */
      err = DB_GET_FIELD_LOCAL (next, ta, copy + (ofs - (psaddr_t) 0), list_t,
				next, 0);
      if (err != TD_OK)
	break;
    }

  return err;
}


td_err_e
td_ta_thr_iter (const td_thragent_t *ta_arg, td_thr_iter_f *callback,
		void *cbdata_p, td_thr_state_e state, int ti_pri,
		sigset_t *ti_sigmask_p, unsigned int ti_user_flags)
{
  td_thragent_t *const ta = (td_thragent_t *) ta_arg;
  td_err_e err;
  psaddr_t list = 0;

  LOG ("td_ta_thr_iter");

  /* Test whether the TA parameter is ok.  */
  if (! ta_ok (ta))
    return TD_BADTA;

  /* The thread library keeps two lists for the running threads.  One
     list contains the thread which are using user-provided stacks
     (this includes the main thread) and the other includes the
     threads for which the thread library allocated the stacks.  We
     have to iterate over both lists separately.  We start with the
     list of threads with user-defined stacks.  */

  err = __td_ta_stack_user (ta, &list);
  if (err == TD_OK)
    err = iterate_thread_list (ta, callback, cbdata_p, state, ti_pri,
			       list, true);

  /* And the threads with stacks allocated by the implementation.  */
  if (err == TD_OK)
    err = __td_ta_stack_used (ta, &list);
  if (err == TD_OK)
    err = iterate_thread_list (ta, callback, cbdata_p, state, ti_pri,
			       list, false);

  return err;
}
