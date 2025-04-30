/* Locate TLS data for a thread.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include "thread_dbP.h"
#include <link.h>

/* Get the DTV slotinfo list head entry from the dynamic loader state
   into *LISTHEAD.  */
static td_err_e
dtv_slotinfo_list (td_thragent_t *ta,
		   psaddr_t *listhead)
{
  td_err_e err;
  psaddr_t head;

  if (__td_ta_rtld_global (ta))
    {
      err = DB_GET_FIELD (head, ta, ta->ta_addr__rtld_global,
			  rtld_global, _dl_tls_dtv_slotinfo_list, 0);
      if (err != TD_OK)
	return err;
    }
  else
    {
      if (ta->ta_addr__dl_tls_dtv_slotinfo_list == 0
	  && td_mod_lookup (ta->ph, NULL, SYM__dl_tls_dtv_slotinfo_list,
			    &ta->ta_addr__dl_tls_dtv_slotinfo_list) != PS_OK)
	return TD_ERR;

      err = _td_fetch_value (ta, ta->ta_var__dl_tls_dtv_slotinfo_list,
			     SYM_DESC__dl_tls_dtv_slotinfo_list,
			     0, ta->ta_addr__dl_tls_dtv_slotinfo_list, &head);
      if (err != TD_OK)
	return err;
    }

  *listhead = head;
  return TD_OK;
}

/* Get the address of the DTV slotinfo entry for MODID into
   *DTVSLOTINFO.  */
static td_err_e
dtv_slotinfo (td_thragent_t *ta,
	      unsigned long int modid,
	      psaddr_t *dtvslotinfo)
{
  td_err_e err;
  psaddr_t slot, temp;
  size_t slbase = 0;

  err = dtv_slotinfo_list (ta, &slot);
  if (err != TD_OK)
    return err;

  while (slot)
    {
      /* Get the number of entries in this list entry's array.  */
      err = DB_GET_FIELD (temp, ta, slot, dtv_slotinfo_list, len, 0);
      if (err != TD_OK)
	return err;
      size_t len = (uintptr_t)temp;

      /* Did we find the list entry for modid?  */
      if (modid < slbase + len)
	break;

      /* We didn't, so get the next list entry.  */
      slbase += len;
      err = DB_GET_FIELD (temp, ta, slot, dtv_slotinfo_list,
			  next, 0);
      if (err != TD_OK)
	return err;
      slot = temp;
    }

  /* We reached the end of the list and found nothing.  */
  if (!slot)
    return TD_ERR;

  /* Take the slotinfo for modid from the list entry.  */
  err = DB_GET_FIELD_ADDRESS (temp, ta, slot, dtv_slotinfo_list,
			      slotinfo, modid - slbase);
  if (err != TD_OK)
    return err;
  slot = temp;

  *dtvslotinfo = slot;
  return TD_OK;
}

/* Return in *BASE the base address of the TLS block for MODID within
   TH.

   It should return success and yield the correct pointer in any
   circumstance where the TLS block for the module and thread
   requested has already been initialized.

   It should fail with TD_TLSDEFER only when the thread could not
   possibly have observed any values in that TLS block.  That way, the
   debugger can fall back to showing initial values from the PT_TLS
   segment (and refusing attempts to mutate) for the TD_TLSDEFER case,
   and never fail to make the values the program will actually see
   available to the user of the debugger.  */
td_err_e
td_thr_tlsbase (const td_thrhandle_t *th,
		unsigned long int modid,
		psaddr_t *base)
{
  td_err_e err;
  psaddr_t dtv, dtvslot, dtvptr, temp;

  if (modid < 1)
    return TD_NOTLS;

  psaddr_t pd = th->th_unique;
  if (pd == 0)
    {
      /* This is the fake handle for the main thread before libpthread
	 initialization.  We are using 0 for its th_unique because we can't
	 trust that its thread register has been initialized.  But we need
	 a real pointer to have any TLS access work.  In case of dlopen'd
	 libpthread, initialization might not be for quite some time.  So
	 try looking up the thread register now.  Worst case, it's nonzero
	 uninitialized garbage and we get bogus results for TLS access
	 attempted too early.  Tough.  */

      td_thrhandle_t main_th;
      err = __td_ta_lookup_th_unique (th->th_ta_p, ps_getpid (th->th_ta_p->ph),
				      &main_th);
      if (err == 0)
	pd = main_th.th_unique;
      if (pd == 0)
	return TD_TLSDEFER;
    }

  err = dtv_slotinfo (th->th_ta_p, modid, &temp);
  if (err != TD_OK)
    return err;

  psaddr_t slot;
  err = DB_GET_STRUCT (slot, th->th_ta_p, temp, dtv_slotinfo);
  if (err != TD_OK)
    return err;

  /* Take the link_map from the slotinfo.  */
  psaddr_t map;
  err = DB_GET_FIELD_LOCAL (map, th->th_ta_p, slot, dtv_slotinfo, map, 0);
  if (err != TD_OK)
    return err;
  if (!map)
    return TD_ERR;

  /* Ok, the modid is good, now find out what DTV generation it
     requires.  */
  err = DB_GET_FIELD_LOCAL (temp, th->th_ta_p, slot, dtv_slotinfo, gen, 0);
  if (err != TD_OK)
    return err;
  size_t modgen = (uintptr_t)temp;

  /* Get the DTV pointer from the thread descriptor.  */
  err = DB_GET_FIELD (dtv, th->th_ta_p, pd, pthread, dtvp, 0);
  if (err != TD_OK)
    return err;

  psaddr_t dtvgenloc;
  /* Get the DTV generation count at dtv[0].counter.  */
  err = DB_GET_FIELD_ADDRESS (dtvgenloc, th->th_ta_p, dtv, dtv, dtv, 0);
  if (err != TD_OK)
    return err;
  err = DB_GET_FIELD (temp, th->th_ta_p, dtvgenloc, dtv_t, counter, 0);
  if (err != TD_OK)
    return err;
  size_t dtvgen = (uintptr_t)temp;

  /* Is the DTV current enough?  */
  if (dtvgen < modgen)
    {
    try_static_tls:
      /* If the module uses Static TLS, we're still good.  */
      err = DB_GET_FIELD (temp, th->th_ta_p, map, link_map, l_tls_offset, 0);
      if (err != TD_OK)
	return err;
      ptrdiff_t tlsoff = (uintptr_t)temp;

      if (tlsoff != FORCED_DYNAMIC_TLS_OFFSET
	  && tlsoff != NO_TLS_OFFSET)
	{
	  psaddr_t tp = pd;

#if TLS_TCB_AT_TP
	  dtvptr = tp - tlsoff;
#elif TLS_DTV_AT_TP
	  dtvptr = tp + tlsoff + TLS_PRE_TCB_SIZE;
#else
# error "Either TLS_TCB_AT_TP or TLS_DTV_AT_TP must be defined"
#endif

	  *base = dtvptr;
	  return TD_OK;
	}

      return TD_TLSDEFER;
    }

  /* Find the corresponding entry in the DTV.  */
  err = DB_GET_FIELD_ADDRESS (dtvslot, th->th_ta_p, dtv, dtv, dtv, modid);
  if (err != TD_OK)
    return err;

  /* Extract the TLS block address from that DTV slot.  */
  err = DB_GET_FIELD (dtvptr, th->th_ta_p, dtvslot, dtv_t, pointer_val, 0);
  if (err != TD_OK)
    return err;

  /* It could be that the memory for this module is not allocated for
     the given thread.  */
  if ((uintptr_t) dtvptr & 1)
    goto try_static_tls;

  *base = dtvptr;
  return TD_OK;
}
