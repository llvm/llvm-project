/* List of symbols in libpthread examined by libthread_db.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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

#define DOT(x)	x		/* No prefix.  */

#define STRINGIFY(name)		STRINGIFY_1(name)
#define STRINGIFY_1(name)	#name

#define DB_STRUCT(type)	\
  DB_LOOKUP_NAME (SYM_SIZEOF_##type, _thread_db_sizeof_##type)
#define DB_STRUCT_FIELD(type, field) \
  DB_LOOKUP_NAME (SYM_##type##_FIELD_##field, _thread_db_##type##_##field)
#define DB_STRUCT_FLEXIBLE_ARRAY(type, field) DB_STRUCT_FIELD (type, field)
#define DB_SYMBOL(name) \
  DB_LOOKUP_NAME (SYM_##name, name)
#define DB_FUNCTION(name) \
  DB_LOOKUP_NAME (SYM_##name, DOT (name))
#define DB_VARIABLE(name) \
  DB_LOOKUP_NAME (SYM_##name, name) \
  DB_LOOKUP_NAME (SYM_DESC_##name, _thread_db_##name)

# include "structs.def"

# undef DB_STRUCT
# undef DB_STRUCT_FIELD
# undef DB_STRUCT_FLEXIBLE_ARRAY
# undef DB_FUNCTION
# undef DB_SYMBOL
# undef DB_VARIABLE
# undef DOT

DB_LOOKUP_NAME_TH_UNIQUE (SYM_TH_UNIQUE_REGISTER64, _thread_db_register64)
DB_LOOKUP_NAME_TH_UNIQUE (SYM_TH_UNIQUE_REGISTER32, _thread_db_register32)
DB_LOOKUP_NAME_TH_UNIQUE (SYM_TH_UNIQUE_CONST_THREAD_AREA,
			  _thread_db_const_thread_area)
DB_LOOKUP_NAME_TH_UNIQUE (SYM_TH_UNIQUE_REGISTER32_THREAD_AREA,
			  _thread_db_register32_thread_area)
DB_LOOKUP_NAME_TH_UNIQUE (SYM_TH_UNIQUE_REGISTER64_THREAD_AREA,
			  _thread_db_register64_thread_area)
