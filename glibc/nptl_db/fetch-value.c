/* Helper routines for libthread_db.
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
#include <byteswap.h>
#include <assert.h>
#include <stdint.h>

td_err_e
_td_check_sizeof (td_thragent_t *ta, uint32_t *sizep, int sizep_name)
{
  if (*sizep == 0)
    {
      psaddr_t descptr;
      ps_err_e err = td_lookup (ta->ph, sizep_name, &descptr);
      if (err == PS_NOSYM)
	return TD_NOCAPAB;
      if (err == PS_OK)
	err = ps_pdread (ta->ph, descptr, sizep, sizeof *sizep);
      if (err != PS_OK)
	return TD_ERR;
      if (*sizep & 0xff000000U)
	*sizep = bswap_32 (*sizep);
    }
  return TD_OK;
}

td_err_e
_td_locate_field (td_thragent_t *ta,
		  db_desc_t desc, int descriptor_name,
		  psaddr_t idx, psaddr_t *address)
{
  uint32_t elemsize;

  if (DB_DESC_SIZE (desc) == 0)
    {
      /* Read the information about this field from the inferior.  */
      psaddr_t descptr;
      ps_err_e err = td_lookup (ta->ph, descriptor_name, &descptr);
      if (err == PS_NOSYM)
	return TD_NOCAPAB;
      if (err == PS_OK)
	err = ps_pdread (ta->ph, descptr, desc, DB_SIZEOF_DESC);
      if (err != PS_OK)
	return TD_ERR;
      if (DB_DESC_SIZE (desc) == 0)
	return TD_DBERR;
      if (DB_DESC_SIZE (desc) & 0xff000000U)
	{
	  /* Byte-swap these words, though we leave the size word
	     in native order as the handy way to distinguish.  */
	  DB_DESC_OFFSET (desc) = bswap_32 (DB_DESC_OFFSET (desc));
	  DB_DESC_NELEM (desc) = bswap_32 (DB_DESC_NELEM (desc));
	}
    }

  if (idx != 0 && DB_DESC_NELEM (desc) != 0
      && idx - (psaddr_t) 0 > DB_DESC_NELEM (desc))
    /* This is an internal indicator to callers with nonzero IDX
       that the IDX value is too big.  */
    return TD_NOAPLIC;

  elemsize = DB_DESC_SIZE (desc);
  if (elemsize & 0xff000000U)
    elemsize = bswap_32 (elemsize);

  *address += (int32_t) DB_DESC_OFFSET (desc);
  *address += (elemsize / 8 * (idx - (psaddr_t) 0));
  return TD_OK;
}

td_err_e
_td_fetch_value (td_thragent_t *ta,
		 db_desc_t desc, int descriptor_name,
		 psaddr_t idx, psaddr_t address,
		 psaddr_t *result)
{
  ps_err_e err;
  td_err_e terr = _td_locate_field (ta, desc, descriptor_name, idx, &address);
  if (terr != TD_OK)
    return terr;

  if (DB_DESC_SIZE (desc) == 8 || DB_DESC_SIZE (desc) == bswap_32 (8))
    {
      uint8_t value;
      err = ps_pdread (ta->ph, address, &value, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == 32)
    {
      uint32_t value;
      err = ps_pdread (ta->ph, address, &value, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == 64)
    {
      uint64_t value;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      err = ps_pdread (ta->ph, address, &value, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (32))
    {
      uint32_t value;
      err = ps_pdread (ta->ph, address, &value, sizeof value);
      value = bswap_32 (value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (64))
    {
      uint64_t value;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      err = ps_pdread (ta->ph, address, &value, sizeof value);
      value = bswap_64 (value);
      *result = (psaddr_t) 0 + value;
    }
  else
    return TD_DBERR;

  return err == PS_OK ? TD_OK : TD_ERR;
}


td_err_e
_td_store_value (td_thragent_t *ta,
		 db_desc_t desc, int descriptor_name, psaddr_t idx,
		 psaddr_t address, psaddr_t widened_value)
{
  ps_err_e err;
  td_err_e terr = _td_locate_field (ta, desc, descriptor_name, idx, &address);
  if (terr != TD_OK)
    return terr;

  if (DB_DESC_SIZE (desc) == 8 || DB_DESC_SIZE (desc) == bswap_32 (8))
    {
      uint8_t value = widened_value - (psaddr_t) 0;
      err = ps_pdwrite (ta->ph, address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == 32)
    {
      uint32_t value = widened_value - (psaddr_t) 0;
      err = ps_pdwrite (ta->ph, address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == 64)
    {
      uint64_t value = widened_value - (psaddr_t) 0;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      err = ps_pdwrite (ta->ph, address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (32))
    {
      uint32_t value = widened_value - (psaddr_t) 0;
      value = bswap_32 (value);
      err = ps_pdwrite (ta->ph, address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (64))
    {
      uint64_t value = widened_value - (psaddr_t) 0;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      value = bswap_64 (value);
      err = ps_pdwrite (ta->ph, address, &value, sizeof value);
    }
  else
    return TD_DBERR;

  return err == PS_OK ? TD_OK : TD_ERR;
}

td_err_e
_td_fetch_value_local (td_thragent_t *ta,
		       db_desc_t desc, int descriptor_name, psaddr_t idx,
		       void *address,
		       psaddr_t *result)
{
  td_err_e terr = _td_locate_field (ta, desc, descriptor_name, idx, &address);
  if (terr != TD_OK)
    return terr;

  if (DB_DESC_SIZE (desc) == 8 || DB_DESC_SIZE (desc) == bswap_32 (8))
    {
      uint8_t value;
      memcpy (&value, address, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == 32)
    {
      uint32_t value;
      memcpy (&value, address, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == 64)
    {
      uint64_t value;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      memcpy (&value, address, sizeof value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (32))
    {
      uint32_t value;
      memcpy (&value, address, sizeof value);
      value = bswap_32 (value);
      *result = (psaddr_t) 0 + value;
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (64))
    {
      uint64_t value;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      memcpy (&value, address, sizeof value);
      value = bswap_64 (value);
      *result = (psaddr_t) 0 + value;
    }
  else
    return TD_DBERR;

  return TD_OK;
}


td_err_e
_td_store_value_local (td_thragent_t *ta,
		       db_desc_t desc, int descriptor_name, psaddr_t idx,
		       void *address, psaddr_t widened_value)
{
  td_err_e terr = _td_locate_field (ta, desc, descriptor_name, idx, &address);
  if (terr != TD_OK)
    return terr;

  if (DB_DESC_SIZE (desc) == 8 || DB_DESC_SIZE (desc) == bswap_32 (8))
    {
      uint8_t value = widened_value - (psaddr_t) 0;
      memcpy (address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == 32)
    {
      uint32_t value = widened_value - (psaddr_t) 0;
      memcpy (address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == 64)
    {
      uint64_t value = widened_value - (psaddr_t) 0;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      memcpy (address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (32))
    {
      uint32_t value = widened_value - (psaddr_t) 0;
      value = bswap_32 (value);
      memcpy (address, &value, sizeof value);
    }
  else if (DB_DESC_SIZE (desc) == bswap_32 (64))
    {
      uint64_t value = widened_value - (psaddr_t) 0;
      if (sizeof (psaddr_t) < 8)
	return TD_NOCAPAB;
      value = bswap_64 (value);
      memcpy (address, &value, sizeof value);
    }
  else
    return TD_DBERR;

  return TD_OK;
}
