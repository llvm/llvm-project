/* This file is included by pthread_create.c to define in libpthread
   all the magic symbols required by libthread_db.

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

#include <stdint.h>
#include "thread_dbP.h"
#include <tls.h>
#include <ldsodefs.h>

typedef struct pthread pthread;
typedef struct pthread_key_struct pthread_key_struct;
typedef struct pthread_key_data pthread_key_data;
typedef struct
{
  struct pthread_key_data data[PTHREAD_KEY_2NDLEVEL_SIZE];
}
pthread_key_data_level2;

typedef struct
{
  union dtv dtv[UINT32_MAX / 2 / sizeof (union dtv)]; /* No constant bound.  */
} dtv;

typedef struct link_map link_map;
typedef struct rtld_global rtld_global;
typedef struct dtv_slotinfo_list dtv_slotinfo_list;
typedef struct dtv_slotinfo dtv_slotinfo;

#define schedparam_sched_priority schedparam.sched_priority

#define eventbuf_eventmask eventbuf.eventmask
#define eventbuf_eventmask_event_bits eventbuf.eventmask.event_bits

#define DESC(name, offset, obj) \
  DB_DEFINE_DESC (name, 8 * sizeof (obj), 1, offset);
#define ARRAY_DESC(name, offset, obj) \
  DB_DEFINE_DESC (name, \
		  8 * sizeof (obj)[0], sizeof (obj) / sizeof (obj)[0], \
		  offset);
/* Flexible arrays do not have a length that can be determined.  */
#define FLEXIBLE_ARRAY_DESC(name, offset, obj) \
  DB_DEFINE_DESC (name, 8 * sizeof (obj)[0], 0, offset);

#if TLS_TCB_AT_TP
# define dtvp header.dtv
#elif TLS_DTV_AT_TP
/* Special case hack.  If TLS_TCB_SIZE == 0 (on PowerPC), there is no TCB
   containing the DTV at the TP, but actually the TCB lies behind the TP,
   i.e. at the very end of the area covered by TLS_PRE_TCB_SIZE.  */
DESC (_thread_db_pthread_dtvp,
      TLS_PRE_TCB_SIZE + offsetof (tcbhead_t, dtv)
      - (TLS_TCB_SIZE == 0 ? sizeof (tcbhead_t) : 0), union dtv *)
#endif


#define DB_STRUCT(type) \
  const uint32_t _thread_db_sizeof_##type = sizeof (type);
#define DB_STRUCT_FIELD(type, field) \
  DESC (_thread_db_##type##_##field, \
	offsetof (type, field), ((type *) 0)->field)
#define DB_STRUCT_ARRAY_FIELD(type, field) \
  ARRAY_DESC (_thread_db_##type##_##field, \
	      offsetof (type, field), ((type *) 0)->field)
#define DB_STRUCT_FLEXIBLE_ARRAY(type, field) \
  FLEXIBLE_ARRAY_DESC (_thread_db_##type##_##field, \
		       offsetof (type, field), ((type *) 0)->field)
#define DB_VARIABLE(name) DESC (_thread_db_##name, 0, name)
#define DB_ARRAY_VARIABLE(name) ARRAY_DESC (_thread_db_##name, 0, name)
#define DB_SYMBOL(name)	/* Nothing.  */
#define DB_FUNCTION(name) /* Nothing.  */
#include "structs.def"
#undef DB_STRUCT
#undef DB_STRUCT_FIELD
#undef DB_SYMBOL
#undef DB_FUNCTION
#undef DB_VARIABLE
#undef DESC



#ifdef DB_THREAD_SELF
# ifdef DB_THREAD_SELF_INCLUDE
#  include DB_THREAD_SELF_INCLUDE
# endif

/* This macro is defined in the machine's tls.h using the three below.  */
# define CONST_THREAD_AREA(bits, value) \
  const uint32_t _thread_db_const_thread_area = (value);
# define REGISTER_THREAD_AREA(bits, regofs, scale) \
  DB_DEFINE_DESC (_thread_db_register##bits##_thread_area, \
		  bits, (scale), (regofs));
# define REGISTER(bits, size, regofs, bias) \
  DB_DEFINE_DESC (_thread_db_register##bits, size, (uint32_t)(bias), (regofs));

DB_THREAD_SELF
#endif
