/* Helper functions used by strftime/strptime to handle locale-specific "eras".
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include "../locale/localeinfo.h"
#include <libc-lock.h>
#include <stdlib.h>
#include <wchar.h>
#include <string.h>
#include <stdint.h>

/* Some of the functions here must not be used while setlocale is called.  */
__libc_rwlock_define (extern, __libc_setlocale_lock attribute_hidden)

#define CURRENT(item)		(current->values[_NL_ITEM_INDEX (item)].string)
#define CURRENT_WORD(item)	(current->values[_NL_ITEM_INDEX (item)].word)

#define ERA_DATE_CMP(a, b) \
  (a[0] < b[0] || (a[0] == b[0] && (a[1] < b[1]				      \
				    || (a[1] == b[1] && a[2] <= b[2]))))

/* Look up the era information in CURRENT's locale strings and
   cache it in CURRENT->private.  */
static void
_nl_init_era_entries (struct __locale_data *current)
{
  size_t cnt;
  struct lc_time_data *data;

  /* Avoid touching CURRENT if there is no data at all, for _nl_C_LC_TIME.  */
  if (CURRENT_WORD (_NL_TIME_ERA_NUM_ENTRIES) == 0)
    return;

  __libc_rwlock_wrlock (__libc_setlocale_lock);

  if (current->private.time == NULL)
    {
      current->private.time = malloc (sizeof *current->private.time);
      if (current->private.time == NULL)
	goto out;
      memset (current->private.time, 0, sizeof *current->private.time);
      current->private.cleanup = &_nl_cleanup_time;
    }
  data = current->private.time;

  if (! data->era_initialized)
    {
      size_t new_num_eras = CURRENT_WORD (_NL_TIME_ERA_NUM_ENTRIES);
      if (new_num_eras == 0)
	{
	  if (data->eras != NULL)
	    {
	      free (data->eras);
	      data->eras = NULL;
	    }
	}
      else
	{
	  struct era_entry *new_eras = data->eras;

	  if (data->num_eras != new_num_eras)
	    new_eras =
	      (struct era_entry *) realloc (data->eras,
					    new_num_eras
					    * sizeof (struct era_entry));
	  if (new_eras == NULL)
	    {
	      free (data->eras);
	      data->num_eras = 0;
	      data->eras = NULL;
	    }
	  else
	    {
	      const char *ptr = CURRENT (_NL_TIME_ERA_ENTRIES);
	      data->num_eras = new_num_eras;
	      data->eras = new_eras;

	      for (cnt = 0; cnt < new_num_eras; ++cnt)
		{
		  const char *base_ptr = ptr;
		  memcpy ((void *) (new_eras + cnt), (const void *) ptr,
			  sizeof (uint32_t) * 8);

		  if (ERA_DATE_CMP(new_eras[cnt].start_date,
				   new_eras[cnt].stop_date))
		    if (new_eras[cnt].direction == (uint32_t) '+')
		      new_eras[cnt].absolute_direction = 1;
		    else
		      new_eras[cnt].absolute_direction = -1;
		  else
		    if (new_eras[cnt].direction == (uint32_t) '+')
		      new_eras[cnt].absolute_direction = -1;
		    else
		      new_eras[cnt].absolute_direction = 1;

		  /* Skip numeric values.  */
		  ptr += sizeof (uint32_t) * 8;

		  /* Set and skip era name.  */
		  new_eras[cnt].era_name = ptr;
		  ptr = strchr (ptr, '\0') + 1;

		  /* Set and skip era format.  */
		  new_eras[cnt].era_format = ptr;
		  ptr = strchr (ptr, '\0') + 1;

		  ptr += 3 - (((ptr - (const char *) base_ptr) + 3) & 3);

		  /* Set and skip wide era name.  */
		  new_eras[cnt].era_wname = (wchar_t *) ptr;
		  ptr = (char *) (__wcschr ((wchar_t *) ptr, L'\0') + 1);

		  /* Set and skip wide era format.  */
		  new_eras[cnt].era_wformat = (wchar_t *) ptr;
		  ptr = (char *) (__wcschr ((wchar_t *) ptr, L'\0') + 1);
		}
	    }
	}

      data->era_initialized = 1;
    }

 out:
  __libc_rwlock_unlock (__libc_setlocale_lock);
}

struct era_entry *
_nl_get_era_entry (const struct tm *tp, struct __locale_data *current)
{
  if (current->private.time == NULL || !current->private.time->era_initialized)
    _nl_init_era_entries (current);

  if (current->private.time != NULL)
    {
      /* Now compare date with the available eras.  */
      const int32_t tdate[3] = { tp->tm_year, tp->tm_mon, tp->tm_mday };
      size_t cnt;
      for (cnt = 0; cnt < current->private.time->num_eras; ++cnt)
	if ((ERA_DATE_CMP (current->private.time->eras[cnt].start_date, tdate)
	     && ERA_DATE_CMP (tdate,
			      current->private.time->eras[cnt].stop_date))
	    || (ERA_DATE_CMP (current->private.time->eras[cnt].stop_date,
			      tdate)
		&& ERA_DATE_CMP (tdate,
				 current->private.time->eras[cnt].start_date)))
	  return &current->private.time->eras[cnt];
    }

  return NULL;
}


struct era_entry *
_nl_select_era_entry (int cnt, struct __locale_data *current)
{
  if (current->private.time == NULL || !current->private.time->era_initialized)
    _nl_init_era_entries (current);

  return (current->private.time == NULL
	  ? NULL : &current->private.time->eras[cnt]);
}
