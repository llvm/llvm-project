/* Private header for thread debug library
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

#ifndef _THREAD_DBP_H
#define _THREAD_DBP_H	1

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include "proc_service.h"
#include "thread_db.h"
#include <pthreadP.h>  	/* This is for *_BITMASK only.  */
#include <list.h>
#include <gnu/lib-names.h>
#include <libc-diag.h>

/* Indeces for the symbol names.  */
enum
  {
# define DB_STRUCT(type)		SYM_SIZEOF_##type,
# define DB_STRUCT_FIELD(type, field)	SYM_##type##_FIELD_##field,
# define DB_STRUCT_FLEXIBLE_ARRAY(type, field) DB_STRUCT_FIELD (type, field)
# define DB_SYMBOL(name)		SYM_##name,
# define DB_FUNCTION(name)		SYM_##name,
# define DB_VARIABLE(name)		SYM_##name, SYM_DESC_##name,
# include "structs.def"
# undef DB_STRUCT
# undef DB_STRUCT_FIELD
# undef DB_STRUCT_FLEXIBLE_ARRAY
# undef DB_SYMBOL
# undef DB_FUNCTION
# undef DB_VARIABLE

    SYM_TH_UNIQUE_CONST_THREAD_AREA,
    SYM_TH_UNIQUE_REGISTER64,
    SYM_TH_UNIQUE_REGISTER32,
    SYM_TH_UNIQUE_REGISTER64_THREAD_AREA,
    SYM_TH_UNIQUE_REGISTER32_THREAD_AREA,

    SYM_NUM_MESSAGES
  };


/* Comment out the following for less verbose output.  */
#ifndef NDEBUG
# define LOG(c) if (__td_debug) write (2, c "\n", strlen (c "\n"))
extern int __td_debug attribute_hidden;
#else
# define LOG(c)
#endif


#define DB_DESC_SIZE(desc)	((desc)[0])
#define DB_DESC_NELEM(desc)	((desc)[1])
#define DB_DESC_OFFSET(desc)	((desc)[2])
#define DB_SIZEOF_DESC		(3 * sizeof (uint32_t))
#define DB_DEFINE_DESC(name, size, nelem, offset) \
  const uint32_t name[3] = { (size), (nelem), (offset) }
typedef uint32_t db_desc_t[3];


/* Handle for a process.  This type is opaque.  */
struct td_thragent
{
  /* Chain on the list of all agent structures.  */
  list_t list;

  /* Delivered by the debugger and we have to pass it back in the
     proc callbacks.  */
  struct ps_prochandle *ph;

  /* Cached values read from the inferior.  */
# define DB_STRUCT(type) \
  uint32_t ta_sizeof_##type;
# define DB_STRUCT_FIELD(type, field) \
  db_desc_t ta_field_##type##_##field;
# define DB_STRUCT_FLEXIBLE_ARRAY(type, field) DB_STRUCT_FIELD (type, field)
# define DB_SYMBOL(name) \
  psaddr_t ta_addr_##name;
# define DB_FUNCTION(name) \
  psaddr_t ta_addr_##name;
# define DB_VARIABLE(name) \
  psaddr_t ta_addr_##name; \
  db_desc_t ta_var_##name;
# include "structs.def"
# undef DB_STRUCT
# undef DB_STRUCT_FIELD
# undef DB_STRUCT_FLEXIBLE_ARRAY
# undef DB_FUNCTION
# undef DB_SYMBOL
# undef DB_VARIABLE

  psaddr_t ta_addr__rtld_global;

  /* The method of locating a thread's th_unique value.  */
  enum
    {
      ta_howto_unknown,
      ta_howto_reg,
      ta_howto_reg_thread_area,
      ta_howto_const_thread_area
    } ta_howto;
  union
  {
    uint32_t const_thread_area;	/* Constant argument to ps_get_thread_area.  */
    /* These are as if the descriptor of the field in prregset_t,
       but DB_DESC_NELEM is overloaded as follows: */
    db_desc_t reg;		/* Signed bias applied to register value.  */
    db_desc_t reg_thread_area;	/* Bits to scale down register value.  */
  } ta_howto_data;
};


/* List of all known descriptors.  */
extern list_t __td_agent_list attribute_hidden;


/* Function used to test for correct thread agent pointer.  */
static inline bool
ta_ok (const td_thragent_t *ta)
{
  list_t *runp;

  list_for_each (runp, &__td_agent_list)
    if (list_entry (runp, td_thragent_t, list) == ta)
      return true;

  return false;
}


/* Internal wrappers around ps_pglobal_lookup.  */
extern ps_err_e td_mod_lookup (struct ps_prochandle *ps, const char *modname,
			       int idx, psaddr_t *sym_addr) attribute_hidden;
#define td_lookup(ps, idx, sym_addr) \
  td_mod_lookup ((ps), LIBPTHREAD_SO, (idx), (sym_addr))


/* Store in psaddr_t VAR the address of inferior's symbol NAME.  */
#define DB_GET_SYMBOL(var, ta, name)					      \
  (((ta)->ta_addr_##name == 0						      \
    && td_lookup ((ta)->ph, SYM_##name, &(ta)->ta_addr_##name) != PS_OK)      \
   ? TD_ERR : ((var) = (ta)->ta_addr_##name, TD_OK))

/* Store in psaddr_t VAR the value of ((TYPE) PTR)->FIELD[IDX] in the inferior.
   A target field smaller than psaddr_t is zero-extended.  */
#define DB_GET_FIELD(var, ta, ptr, type, field, idx) \
  _td_fetch_value ((ta), (ta)->ta_field_##type##_##field, \
		   SYM_##type##_FIELD_##field, \
		   (psaddr_t) 0 + (idx), (ptr), &(var))

/* With GCC 5.3 when compiling with -Os the compiler emits a warning
   that slot may be used uninitialized.  This is never the case since
   the dynamic loader initializes the slotinfo list and
   dtv_slotinfo_list will point slot at the first entry.  Therefore
   when DB_GET_FIELD_ADDRESS is called with a slot for ptr, the slot is
   always initialized.  */
DIAG_PUSH_NEEDS_COMMENT;
DIAG_IGNORE_Os_NEEDS_COMMENT (5, "-Wmaybe-uninitialized");
#define DB_GET_FIELD_ADDRESS(var, ta, ptr, type, field, idx) \
  ((var) = (ptr), _td_locate_field ((ta), (ta)->ta_field_##type##_##field, \
				    SYM_##type##_FIELD_##field, \
				    (psaddr_t) 0 + (idx), &(var)))
DIAG_POP_NEEDS_COMMENT;

extern td_err_e _td_locate_field (td_thragent_t *ta,
				  db_desc_t desc, int descriptor_name,
				  psaddr_t idx,
				  psaddr_t *address) attribute_hidden;


/* Like DB_GET_FIELD, but PTR is a local pointer to a structure that
   has already been copied in from the inferior.  */
#define DB_GET_FIELD_LOCAL(var, ta, ptr, type, field, idx) \
  _td_fetch_value_local ((ta), (ta)->ta_field_##type##_##field, \
		         SYM_##type##_FIELD_##field, \
			 (psaddr_t) 0 + (idx), (ptr), &(var))

/* Store in psaddr_t VAR the value of variable NAME[IDX] in the inferior.
   A target value smaller than psaddr_t is zero-extended.  */
#define DB_GET_VALUE(var, ta, name, idx)				      \
  (((ta)->ta_addr_##name == 0						      \
    && td_lookup ((ta)->ph, SYM_##name, &(ta)->ta_addr_##name) != PS_OK)      \
   ? TD_ERR								      \
   : _td_fetch_value ((ta), (ta)->ta_var_##name, SYM_DESC_##name, 	      \
		      (psaddr_t) 0 + (idx), (ta)->ta_addr_##name, &(var)))

/* Helper functions for those.  */
extern td_err_e _td_fetch_value (td_thragent_t *ta,
				 db_desc_t field, int descriptor_name,
				 psaddr_t idx, psaddr_t address,
				 psaddr_t *result) attribute_hidden;
extern td_err_e _td_fetch_value_local (td_thragent_t *ta,
				       db_desc_t field,
				       int descriptor_name,
				       psaddr_t idx, void *address,
				       psaddr_t *result) attribute_hidden;

/* Store psaddr_t VALUE in ((TYPE) PTR)->FIELD[IDX] in the inferior.
   A target field smaller than psaddr_t is zero-extended.  */
#define DB_PUT_FIELD(ta, ptr, type, field, idx, value) \
  _td_store_value ((ta), (ta)->ta_field_##type##_##field, \
		   SYM_##type##_FIELD_##field, \
		   (psaddr_t) 0 + (idx), (ptr), (value))

#define DB_PUT_FIELD_LOCAL(ta, ptr, type, field, idx, value) \
  _td_store_value_local ((ta), (ta)->ta_field_##type##_##field, \
			 SYM_##type##_FIELD_##field, \
			 (psaddr_t) 0 + (idx), (ptr), (value))

/* Store psaddr_t VALUE in variable NAME[IDX] in the inferior.
   A target field smaller than psaddr_t is zero-extended.  */
#define DB_PUT_VALUE(ta, name, idx, value)				      \
  (((ta)->ta_addr_##name == 0						      \
    && td_lookup ((ta)->ph, SYM_##name, &(ta)->ta_addr_##name) != PS_OK)      \
   ? TD_ERR								      \
   : _td_store_value ((ta), (ta)->ta_var_##name, SYM_DESC_##name, 	      \
		      (psaddr_t) 0 + (idx), (ta)->ta_addr_##name, (value)))

/* Helper functions for those.  */
extern td_err_e _td_store_value (td_thragent_t *ta,
				 db_desc_t field, int descriptor_name,
				 psaddr_t idx, psaddr_t address,
				 psaddr_t value) attribute_hidden;
extern td_err_e _td_store_value_local (td_thragent_t *ta,
				       db_desc_t field, int descriptor_name,
				       psaddr_t idx, void *address,
				       psaddr_t value) attribute_hidden;

#define DB_GET_STRUCT(var, ta, ptr, type)				      \
  ({ td_err_e _err = TD_OK;						      \
     if ((ta)->ta_sizeof_##type == 0)					      \
       _err = _td_check_sizeof ((ta), &(ta)->ta_sizeof_##type,		      \
				      SYM_SIZEOF_##type);		      \
     if (_err == TD_OK)							      \
       _err = ps_pdread ((ta)->ph, (ptr),				      \
			 (var) = __alloca ((ta)->ta_sizeof_##type),	      \
			 (ta)->ta_sizeof_##type)			      \
	 == PS_OK ? TD_OK : TD_ERR;					      \
     else								      \
       (var) = NULL; 							      \
     _err;								      \
  })
#define DB_PUT_STRUCT(ta, ptr, type, copy)				      \
  ({ assert ((ta)->ta_sizeof_##type != 0);				      \
     ps_pdwrite ((ta)->ph, (ptr), (copy), (ta)->ta_sizeof_##type)	      \
       == PS_OK ? TD_OK : TD_ERR;					      \
  })

extern td_err_e _td_check_sizeof (td_thragent_t *ta, uint32_t *sizep,
				  int sizep_name) attribute_hidden;

extern td_err_e __td_ta_lookup_th_unique (const td_thragent_t *ta,
					  lwpid_t lwpid, td_thrhandle_t *th);

/* Try to initialize TA->ta_addr__rtld_global.  Return true on
   success, false on failure (which may be cached).  */
bool __td_ta_rtld_global (td_thragent_t *ta) attribute_hidden;

/* Obtain the address of the list_t fields _dl_stack_user and
   _dl_stack_used in _rtld_global, or fall back to the global
   variables of the same name (to support statically linked
   programs).  */
td_err_e __td_ta_stack_user (td_thragent_t *ta, psaddr_t *plist)
  attribute_hidden;
td_err_e __td_ta_stack_used (td_thragent_t *ta, psaddr_t *plist)
  attribute_hidden;

#endif /* thread_dbP.h */
