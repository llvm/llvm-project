/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#include <fstab.h>
#include <mntent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libc-lock.h>

#define BUFFER_SIZE 0x1fc0

struct fstab_state
{
  FILE *fs_fp;
  char *fs_buffer;
  struct mntent fs_mntres;
  struct fstab fs_ret;
};

static struct fstab_state *fstab_init (int opt_rewind);
static struct mntent *fstab_fetch (struct fstab_state *state);
static struct fstab *fstab_convert (struct fstab_state *state);

static struct fstab_state fstab_state;


int
setfsent (void)
{
  return fstab_init (1) != NULL;
}


struct fstab *
getfsent (void)
{
  struct fstab_state *state;

  state = fstab_init (0);
  if (state == NULL)
    return NULL;
  if (fstab_fetch (state) == NULL)
    return NULL;
  return fstab_convert (state);
}


struct fstab *
getfsspec (const char *name)
{
  struct fstab_state *state;
  struct mntent *m;

  state = fstab_init (1);
  if (state == NULL)
    return NULL;
  while ((m = fstab_fetch (state)) != NULL)
    if (strcmp (m->mnt_fsname, name) == 0)
      return fstab_convert (state);
  return NULL;
}


struct fstab *
getfsfile (const char *name)
{
  struct fstab_state *state;
  struct mntent *m;

  state = fstab_init (1);
  if (state == NULL)
    return NULL;
  while ((m = fstab_fetch (state)) != NULL)
    if (strcmp (m->mnt_dir, name) == 0)
      return fstab_convert (state);
  return NULL;
}


void
endfsent (void)
{
  struct fstab_state *state;

  state = &fstab_state;
  if (state->fs_fp != NULL)
    {
      (void) __endmntent (state->fs_fp);
      state->fs_fp = NULL;
    }
}


static struct fstab_state *
fstab_init (int opt_rewind)
{
  struct fstab_state *state;
  char *buffer;
  FILE *fp;

  state = &fstab_state;

  buffer = state->fs_buffer;
  if (buffer == NULL)
    {
      buffer = (char *) malloc (BUFFER_SIZE);
      if (buffer == NULL)
	return NULL;
      state->fs_buffer = buffer;
    }

  fp = state->fs_fp;
  if (fp != NULL)
    {
      if (opt_rewind)
	rewind (fp);
    }
  else
    {
      fp = __setmntent (_PATH_FSTAB, "r");
      if (fp == NULL)
	return NULL;
      state->fs_fp = fp;
    }

  return state;
}


static struct mntent *
fstab_fetch (struct fstab_state *state)
{
  return __getmntent_r (state->fs_fp, &state->fs_mntres,
			state->fs_buffer, BUFFER_SIZE);
}


static struct fstab *
fstab_convert (struct fstab_state *state)
{
  struct mntent *m;
  struct fstab *f;

  m = &state->fs_mntres;
  f = &state->fs_ret;

  f->fs_spec = m->mnt_fsname;
  f->fs_file = m->mnt_dir;
  f->fs_vfstype = m->mnt_type;
  f->fs_mntops = m->mnt_opts;
  f->fs_type = (__hasmntopt (m, FSTAB_RW) ? FSTAB_RW
		: __hasmntopt (m, FSTAB_RQ) ? FSTAB_RQ
		: __hasmntopt (m, FSTAB_RO) ? FSTAB_RO
		: __hasmntopt (m, FSTAB_SW) ? FSTAB_SW
		: __hasmntopt (m, FSTAB_XX) ? FSTAB_XX
		: "??");
  f->fs_freq = m->mnt_freq;
  f->fs_passno = m->mnt_passno;
  return f;
}


/* Make sure the memory is freed if the programs ends while in
   memory-debugging mode and something actually was allocated.  */
libc_freeres_fn (fstab_free)
{
  char *buffer;

  buffer = fstab_state.fs_buffer;
  free ((void *) buffer);
}
