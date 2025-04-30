/* Wait for process state.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <array_length.h>

#include <support/process_state.h>
#include <support/xstdio.h>
#include <support/check.h>

void
support_process_state_wait (pid_t pid, enum support_process_state state)
{
#ifdef __linux__
  /* For Linux it does a polling check on /proc/<pid>/status checking on
     third field.  */

  /* It mimics the kernel states from fs/proc/array.c  */
  static const struct process_states
  {
    enum support_process_state s;
    char v;
  } process_states[] = {
    { support_process_state_running,      'R' },
    { support_process_state_sleeping,     'S' },
    { support_process_state_disk_sleep,   'D' },
    { support_process_state_stopped,      'T' },
    { support_process_state_tracing_stop, 't' },
    { support_process_state_dead,         'X' },
    { support_process_state_zombie,       'Z' },
    { support_process_state_parked,       'P' },
  };

  char spath[sizeof ("/proc/" + 3) * sizeof (pid_t) + sizeof ("/status") + 1];
  snprintf (spath, sizeof (spath), "/proc/%i/status", pid);

  FILE *fstatus = xfopen (spath, "r");
  char *line = NULL;
  size_t linesiz = 0;

  for (;;)
    {
      char cur_state = -1;
      while (xgetline (&line, &linesiz, fstatus) > 0)
	if (strncmp (line, "State:", strlen ("State:")) == 0)
	  {
	    TEST_COMPARE (sscanf (line, "%*s %c", &cur_state), 1);
	    break;
	  }
      /* Fallback to nanosleep for invalid state.  */
      if (cur_state == -1)
	break;

      for (size_t i = 0; i < array_length (process_states); ++i)
	if (state & process_states[i].s && cur_state == process_states[i].v)
	  {
	    free (line);
	    xfclose (fstatus);
	    return;
	  }

      rewind (fstatus);
      fflush (fstatus);

      if (nanosleep (&(struct timespec) { 0, 10000000 }, NULL) != 0)
	FAIL_EXIT1 ("nanosleep: %m");
    }

  free (line);
  xfclose (fstatus);
  /* Fallback to nanosleep if an invalid state is found.  */
#endif
  nanosleep (&(struct timespec) { 1, 0 }, NULL);
}
