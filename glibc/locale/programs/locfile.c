/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <assert.h>
#include <wchar.h>

#include "../../crypt/md5.h"
#include "localedef.h"
#include "localeinfo.h"
#include "locfile.h"
#include "simple-hash.h"

#include "locfile-kw.h"

#define obstack_chunk_alloc xmalloc
#define obstack_chunk_free free

/* Temporary storage of the locale data before writing it to the archive.  */
static locale_data_t to_archive;


int
locfile_read (struct localedef_t *result, const struct charmap_t *charmap)
{
  const char *filename = result->name;
  const char *repertoire_name = result->repertoire_name;
  int locale_mask = result->needed & ~result->avail;
  struct linereader *ldfile;
  int not_here = ALL_LOCALES;

  /* If no repertoire name was specified use the global one.  */
  if (repertoire_name == NULL)
    repertoire_name = repertoire_global;

  /* Open the locale definition file.  */
  ldfile = lr_open (filename, locfile_hash);
  if (ldfile == NULL)
    {
      if (filename != NULL && filename[0] != '/')
	{
	  char *i18npath = getenv ("I18NPATH");
	  if (i18npath != NULL && *i18npath != '\0')
	    {
	      const size_t pathlen = strlen (i18npath);
	      char i18npathbuf[pathlen + 1];
	      char path[strlen (filename) + 1 + pathlen
			+ sizeof ("/locales/") - 1];
	      char *next;
	      i18npath = memcpy (i18npathbuf, i18npath, pathlen + 1);

	      while (ldfile == NULL
		     && (next = strsep (&i18npath, ":")) != NULL)
		{
		  stpcpy (stpcpy (stpcpy (path, next), "/locales/"), filename);

		  ldfile = lr_open (path, locfile_hash);

		  if (ldfile == NULL)
		    {
		      stpcpy (stpcpy (stpcpy (path, next), "/"), filename);

		      ldfile = lr_open (path, locfile_hash);
		    }
		}
	    }

	  /* Test in the default directory.  */
	  if (ldfile == NULL)
	    {
	      char path[strlen (filename) + 1 + sizeof (LOCSRCDIR)];

	      stpcpy (stpcpy (stpcpy (path, LOCSRCDIR), "/"), filename);
	      ldfile = lr_open (path, locfile_hash);
	    }
	}

      if (ldfile == NULL)
	return 1;
    }

    /* Parse locale definition file and store result in RESULT.  */
  while (1)
    {
      struct token *now = lr_token (ldfile, charmap, NULL, NULL, verbose);
      enum token_t nowtok = now->tok;
      struct token *arg;

      if (nowtok == tok_eof)
	break;

      if (nowtok == tok_eol)
	/* Ignore empty lines.  */
	continue;

      switch (nowtok)
	{
	case tok_escape_char:
	case tok_comment_char:
	  /* We need an argument.  */
	  arg = lr_token (ldfile, charmap, NULL, NULL, verbose);

	  if (arg->tok != tok_ident)
	    {
	      SYNTAX_ERROR (_("bad argument"));
	      continue;
	    }

	  if (arg->val.str.lenmb != 1)
	    {
	      lr_error (ldfile, _("\
argument to `%s' must be a single character"),
			nowtok == tok_escape_char
			? "escape_char" : "comment_char");

	      lr_ignore_rest (ldfile, 0);
	      continue;
	    }

	  if (nowtok == tok_escape_char)
	    ldfile->escape_char = *arg->val.str.startmb;
	  else
	    ldfile->comment_char = *arg->val.str.startmb;
	  break;

	case tok_repertoiremap:
	  /* We need an argument.  */
	  arg = lr_token (ldfile, charmap, NULL, NULL, verbose);

	  if (arg->tok != tok_ident)
	    {
	      SYNTAX_ERROR (_("bad argument"));
	      continue;
	    }

	  if (repertoire_name == NULL)
	    {
	      char *newp = alloca (arg->val.str.lenmb + 1);

	      *((char *) mempcpy (newp, arg->val.str.startmb,
				  arg->val.str.lenmb)) = '\0';
	      repertoire_name = newp;
	    }
	  break;

	case tok_lc_ctype:
	  ctype_read (ldfile, result, charmap, repertoire_name,
		      (locale_mask & CTYPE_LOCALE) == 0);
	  result->avail |= locale_mask & CTYPE_LOCALE;
	  not_here ^= CTYPE_LOCALE;
	  continue;

	case tok_lc_collate:
	  collate_read (ldfile, result, charmap, repertoire_name,
			(locale_mask & COLLATE_LOCALE) == 0);
	  result->avail |= locale_mask & COLLATE_LOCALE;
	  not_here ^= COLLATE_LOCALE;
	  continue;

	case tok_lc_monetary:
	  monetary_read (ldfile, result, charmap, repertoire_name,
			 (locale_mask & MONETARY_LOCALE) == 0);
	  result->avail |= locale_mask & MONETARY_LOCALE;
	  not_here ^= MONETARY_LOCALE;
	  continue;

	case tok_lc_numeric:
	  numeric_read (ldfile, result, charmap, repertoire_name,
			(locale_mask & NUMERIC_LOCALE) == 0);
	  result->avail |= locale_mask & NUMERIC_LOCALE;
	  not_here ^= NUMERIC_LOCALE;
	  continue;

	case tok_lc_time:
	  time_read (ldfile, result, charmap, repertoire_name,
		     (locale_mask & TIME_LOCALE) == 0);
	  result->avail |= locale_mask & TIME_LOCALE;
	  not_here ^= TIME_LOCALE;
	  continue;

	case tok_lc_messages:
	  messages_read (ldfile, result, charmap, repertoire_name,
			 (locale_mask & MESSAGES_LOCALE) == 0);
	  result->avail |= locale_mask & MESSAGES_LOCALE;
	  not_here ^= MESSAGES_LOCALE;
	  continue;

	case tok_lc_paper:
	  paper_read (ldfile, result, charmap, repertoire_name,
		      (locale_mask & PAPER_LOCALE) == 0);
	  result->avail |= locale_mask & PAPER_LOCALE;
	  not_here ^= PAPER_LOCALE;
	  continue;

	case tok_lc_name:
	  name_read (ldfile, result, charmap, repertoire_name,
		     (locale_mask & NAME_LOCALE) == 0);
	  result->avail |= locale_mask & NAME_LOCALE;
	  not_here ^= NAME_LOCALE;
	  continue;

	case tok_lc_address:
	  address_read (ldfile, result, charmap, repertoire_name,
			(locale_mask & ADDRESS_LOCALE) == 0);
	  result->avail |= locale_mask & ADDRESS_LOCALE;
	  not_here ^= ADDRESS_LOCALE;
	  continue;

	case tok_lc_telephone:
	  telephone_read (ldfile, result, charmap, repertoire_name,
			  (locale_mask & TELEPHONE_LOCALE) == 0);
	  result->avail |= locale_mask & TELEPHONE_LOCALE;
	  not_here ^= TELEPHONE_LOCALE;
	  continue;

	case tok_lc_measurement:
	  measurement_read (ldfile, result, charmap, repertoire_name,
			    (locale_mask & MEASUREMENT_LOCALE) == 0);
	  result->avail |= locale_mask & MEASUREMENT_LOCALE;
	  not_here ^= MEASUREMENT_LOCALE;
	  continue;

	case tok_lc_identification:
	  identification_read (ldfile, result, charmap, repertoire_name,
			       (locale_mask & IDENTIFICATION_LOCALE) == 0);
	  result->avail |= locale_mask & IDENTIFICATION_LOCALE;
	  not_here ^= IDENTIFICATION_LOCALE;
	  continue;

	default:
	  SYNTAX_ERROR (_("\
syntax error: not inside a locale definition section"));
	  continue;
	}

      /* The rest of the line must be empty.  */
      lr_ignore_rest (ldfile, 1);
    }

  /* We read all of the file.  */
  lr_close (ldfile);

  /* Mark the categories which are not contained in the file.  We assume
     them to be available and the default data will be used.  */
  result->avail |= not_here;

  return 0;
}


/* Semantic checking of locale specifications.  */

static void (*const check_funcs[]) (struct localedef_t *,
				    const struct charmap_t *) =
{
  [LC_CTYPE] = ctype_finish,
  [LC_COLLATE] = collate_finish,
  [LC_MESSAGES] = messages_finish,
  [LC_MONETARY] = monetary_finish,
  [LC_NUMERIC] = numeric_finish,
  [LC_TIME] = time_finish,
  [LC_PAPER] = paper_finish,
  [LC_NAME] = name_finish,
  [LC_ADDRESS] = address_finish,
  [LC_TELEPHONE] = telephone_finish,
  [LC_MEASUREMENT] = measurement_finish,
  [LC_IDENTIFICATION] = identification_finish
};

void
check_all_categories (struct localedef_t *definitions,
		      const struct charmap_t *charmap)
{
  int cnt;

  for (cnt = 0; cnt < sizeof (check_funcs) / sizeof (check_funcs[0]); ++cnt)
    if (check_funcs[cnt] != NULL)
      check_funcs[cnt] (definitions, charmap);
}


/* Writing the locale data files.  All files use the same output_path.  */

static void (*const write_funcs[]) (struct localedef_t *,
				    const struct charmap_t *, const char *) =
{
  [LC_CTYPE] = ctype_output,
  [LC_COLLATE] = collate_output,
  [LC_MESSAGES] = messages_output,
  [LC_MONETARY] = monetary_output,
  [LC_NUMERIC] = numeric_output,
  [LC_TIME] = time_output,
  [LC_PAPER] = paper_output,
  [LC_NAME] = name_output,
  [LC_ADDRESS] = address_output,
  [LC_TELEPHONE] = telephone_output,
  [LC_MEASUREMENT] = measurement_output,
  [LC_IDENTIFICATION] = identification_output
};


void
write_all_categories (struct localedef_t *definitions,
		      const struct charmap_t *charmap, const char *locname,
		      const char *output_path)
{
  int cnt;

  for (cnt = 0; cnt < sizeof (write_funcs) / sizeof (write_funcs[0]); ++cnt)
    if (write_funcs[cnt] != NULL)
      write_funcs[cnt] (definitions, charmap, output_path);

  if (! no_archive)
    {
      /* The data has to be added to the archive.  Do this now.  */
      struct locarhandle ah;

      /* Open the archive.  This call never returns if we cannot
	 successfully open the archive.  */
      ah.fname = NULL;
      open_archive (&ah, false);

      if (add_locale_to_archive (&ah, locname, to_archive, true) != 0)
	error (EXIT_FAILURE, errno, _("cannot add to locale archive"));

      /* We are done.  */
      close_archive (&ah);
    }
}


/* Return a NULL terminated list of the directories next to output_path
   that have the same owner, group, permissions and device as output_path.  */
static const char **
siblings_uncached (const char *output_path)
{
  size_t len;
  char *base, *p;
  struct stat64 output_stat;
  DIR *dirp;
  int nelems;
  const char **elems;

  /* Remove trailing slashes and trailing pathname component.  */
  len = strlen (output_path);
  base = (char *) alloca (len);
  memcpy (base, output_path, len);
  p = base + len;
  while (p > base && p[-1] == '/')
    p--;
  if (p == base)
    return NULL;
  do
    p--;
  while (p > base && p[-1] != '/');
  if (p == base)
    return NULL;
  *--p = '\0';
  len = p - base;

  /* Get the properties of output_path.  */
  if (lstat64 (output_path, &output_stat) < 0 || !S_ISDIR (output_stat.st_mode))
    return NULL;

  /* Iterate through the directories in base directory.  */
  dirp = opendir (base);
  if (dirp == NULL)
    return NULL;
  nelems = 0;
  elems = NULL;
  for (;;)
    {
      struct dirent64 *other_dentry;
      const char *other_name;
      char *other_path;
      struct stat64 other_stat;

      other_dentry = readdir64 (dirp);
      if (other_dentry == NULL)
	break;

      other_name = other_dentry->d_name;
      if (strcmp (other_name, ".") == 0 || strcmp (other_name, "..") == 0)
	continue;

      other_path = (char *) xmalloc (len + 1 + strlen (other_name) + 2);
      memcpy (other_path, base, len);
      other_path[len] = '/';
      strcpy (other_path + len + 1, other_name);

      if (lstat64 (other_path, &other_stat) >= 0
	  && S_ISDIR (other_stat.st_mode)
	  && other_stat.st_uid == output_stat.st_uid
	  && other_stat.st_gid == output_stat.st_gid
	  && other_stat.st_mode == output_stat.st_mode
	  && other_stat.st_dev == output_stat.st_dev)
	{
	  /* Found a subdirectory.  Add a trailing slash and store it.  */
	  p = other_path + len + 1 + strlen (other_name);
	  *p++ = '/';
	  *p = '\0';
	  elems = (const char **) xrealloc ((char *) elems,
					    (nelems + 2) * sizeof (char **));
	  elems[nelems++] = other_path;
	}
      else
	free (other_path);
    }
  closedir (dirp);

  if (elems != NULL)
    elems[nelems] = NULL;
  return elems;
}


/* Return a NULL terminated list of the directories next to output_path
   that have the same owner, group, permissions and device as output_path.
   Cache the result for future calls.  */
static const char **
siblings (const char *output_path)
{
  static const char *last_output_path;
  static const char **last_result;

  if (output_path != last_output_path)
    {
      if (last_result != NULL)
	{
	  const char **p;

	  for (p = last_result; *p != NULL; p++)
	    free ((char *) *p);
	  free (last_result);
	}

      last_output_path = output_path;
      last_result = siblings_uncached (output_path);
    }
  return last_result;
}


/* Read as many bytes from a file descriptor as possible.  */
static ssize_t
full_read (int fd, void *bufarea, size_t nbyte)
{
  char *buf = (char *) bufarea;

  while (nbyte > 0)
    {
      ssize_t retval = read (fd, buf, nbyte);

      if (retval == 0)
	break;
      else if (retval > 0)
	{
	  buf += retval;
	  nbyte -= retval;
	}
      else if (errno != EINTR)
	return retval;
    }
  return buf - (char *) bufarea;
}


/* Compare the contents of two regular files of the same size.  Return 0
   if they are equal, 1 if they are different, or -1 if an error occurs.  */
static int
compare_files (const char *filename1, const char *filename2, size_t size,
	       size_t blocksize)
{
  int fd1, fd2;
  int ret = -1;

  fd1 = open (filename1, O_RDONLY);
  if (fd1 >= 0)
    {
      fd2 = open (filename2, O_RDONLY);
      if (fd2 >= 0)
	{
	  char *buf1 = (char *) xmalloc (2 * blocksize);
	  char *buf2 = buf1 + blocksize;

	  ret = 0;
	  while (size > 0)
	    {
	      size_t bytes = (size < blocksize ? size : blocksize);

	      if (full_read (fd1, buf1, bytes) < (ssize_t) bytes)
		{
		  ret = -1;
		  break;
		}
	      if (full_read (fd2, buf2, bytes) < (ssize_t) bytes)
		{
		  ret = -1;
		  break;
		}
	      if (memcmp (buf1, buf2, bytes) != 0)
		{
		  ret = 1;
		  break;
		}
	      size -= bytes;
	    }

	  free (buf1);
	  close (fd2);
	}
      close (fd1);
    }
  return ret;
}

/* True if the locale files use the opposite endianness to the
   machine running localedef.  */
bool swap_endianness_p;

/* When called outside a start_locale_structure/end_locale_structure
   or start_locale_prelude/end_locale_prelude block, record that the
   next byte in FILE's obstack will be the first byte of a new element.
   Do likewise for the first call inside a start_locale_structure/
   end_locale_structure block.  */
static void
record_offset (struct locale_file *file)
{
  if (file->structure_stage < 2)
    {
      assert (file->next_element < file->n_elements);
      file->offsets[file->next_element++]
	= (obstack_object_size (&file->data)
	   + (file->n_elements + 2) * sizeof (uint32_t));
      if (file->structure_stage == 1)
	file->structure_stage = 2;
    }
}

/* Initialize FILE for a new output file.  N_ELEMENTS is the number
   of elements in the file.  */
void
init_locale_data (struct locale_file *file, size_t n_elements)
{
  file->n_elements = n_elements;
  file->next_element = 0;
  file->offsets = xmalloc (sizeof (uint32_t) * n_elements);
  obstack_init (&file->data);
  file->structure_stage = 0;
}

/* Align the size of FILE's obstack object to BOUNDARY bytes.  */
void
align_locale_data (struct locale_file *file, size_t boundary)
{
  size_t size = -obstack_object_size (&file->data) & (boundary - 1);
  obstack_blank (&file->data, size);
  memset (obstack_next_free (&file->data) - size, 0, size);
}

/* Record that FILE's next element contains no data.  */
void
add_locale_empty (struct locale_file *file)
{
  record_offset (file);
}

/* Record that FILE's next element consists of SIZE bytes starting at DATA.  */
void
add_locale_raw_data (struct locale_file *file, const void *data, size_t size)
{
  record_offset (file);
  obstack_grow (&file->data, data, size);
}

/* Finish the current object on OBSTACK and use it as the data for FILE's
   next element.  */
void
add_locale_raw_obstack (struct locale_file *file, struct obstack *obstack)
{
  size_t size = obstack_object_size (obstack);
  record_offset (file);
  obstack_grow (&file->data, obstack_finish (obstack), size);
}

/* Use STRING as FILE's next element.  */
void
add_locale_string (struct locale_file *file, const char *string)
{
  record_offset (file);
  obstack_grow (&file->data, string, strlen (string) + 1);
}

/* Likewise for wide strings.  */
void
add_locale_wstring (struct locale_file *file, const uint32_t *string)
{
  add_locale_uint32_array (file, string, wcslen ((const wchar_t *) string) + 1);
}

/* Record that FILE's next element is the 32-bit integer VALUE.  */
void
add_locale_uint32 (struct locale_file *file, uint32_t value)
{
  align_locale_data (file, LOCFILE_ALIGN);
  record_offset (file);
  value = maybe_swap_uint32 (value);
  obstack_grow (&file->data, &value, sizeof (value));
}

/* Record that FILE's next element is an array of N_ELEMS integers
   starting at DATA.  */
void
add_locale_uint32_array (struct locale_file *file,
			 const uint32_t *data, size_t n_elems)
{
  align_locale_data (file, LOCFILE_ALIGN);
  record_offset (file);
  obstack_grow (&file->data, data, n_elems * sizeof (uint32_t));
  maybe_swap_uint32_obstack (&file->data, n_elems);
}

/* Record that FILE's next element is the single byte given by VALUE.  */
void
add_locale_char (struct locale_file *file, char value)
{
  record_offset (file);
  obstack_1grow (&file->data, value);
}

/* Start building an element that contains several different pieces of data.
   Subsequent calls to add_locale_* will add data to the same element up
   till the next call to end_locale_structure.  The element's alignment
   is dictated by the first piece of data added to it.  */
void
start_locale_structure (struct locale_file *file)
{
  assert (file->structure_stage == 0);
  file->structure_stage = 1;
}

/* Finish a structure element that was started by start_locale_structure.
   Empty structures are OK and behave like add_locale_empty.  */
void
end_locale_structure (struct locale_file *file)
{
  record_offset (file);
  assert (file->structure_stage == 2);
  file->structure_stage = 0;
}

/* Start building data that goes before the next element's recorded offset.
   Subsequent calls to add_locale_* will add data to the file without
   treating any of it as the start of a new element.  Calling
   end_locale_prelude switches back to the usual behavior.  */
void
start_locale_prelude (struct locale_file *file)
{
  assert (file->structure_stage == 0);
  file->structure_stage = 3;
}

/* End a block started by start_locale_prelude.  */
void
end_locale_prelude (struct locale_file *file)
{
  assert (file->structure_stage == 3);
  file->structure_stage = 0;
}

/* Write a locale file, with contents given by FILE.  */
void
write_locale_data (const char *output_path, int catidx, const char *category,
		   struct locale_file *file)
{
  size_t cnt, step, maxiov;
  int fd;
  char *fname;
  const char **other_paths = NULL;
  uint32_t header[2];
  size_t n_elem;
  struct iovec vec[3];

  assert (file->n_elements == file->next_element);
  header[0] = LIMAGIC (catidx);
  header[1] = file->n_elements;
  vec[0].iov_len = sizeof (header);
  vec[0].iov_base = header;
  vec[1].iov_len = sizeof (uint32_t) * file->n_elements;
  vec[1].iov_base = file->offsets;
  vec[2].iov_len = obstack_object_size (&file->data);
  vec[2].iov_base = obstack_finish (&file->data);
  maybe_swap_uint32_array (vec[0].iov_base, 2);
  maybe_swap_uint32_array (vec[1].iov_base, file->n_elements);
  n_elem = 3;
  if (! no_archive)
    {
      /* The data will be added to the archive.  For now we simply
	 generate the image which will be written.  First determine
	 the size.  */
      int cnt;
      void *endp;

      to_archive[catidx].size = 0;
      for (cnt = 0; cnt < n_elem; ++cnt)
	to_archive[catidx].size += vec[cnt].iov_len;

      /* Allocate the memory for it.  */
      to_archive[catidx].addr = xmalloc (to_archive[catidx].size);

      /* Fill it in.  */
      for (cnt = 0, endp = to_archive[catidx].addr; cnt < n_elem; ++cnt)
	endp = mempcpy (endp, vec[cnt].iov_base, vec[cnt].iov_len);

      /* Compute the MD5 sum for the data.  */
      __md5_buffer (to_archive[catidx].addr, to_archive[catidx].size,
		    to_archive[catidx].sum);

      return;
    }

  fname = xmalloc (strlen (output_path) + 2 * strlen (category) + 7);

  /* Normally we write to the directory pointed to by the OUTPUT_PATH.
     But for LC_MESSAGES we have to take care for the translation
     data.  This means we need to have a directory LC_MESSAGES in
     which we place the file under the name SYS_LC_MESSAGES.  */
  sprintf (fname, "%s%s", output_path, category);
  fd = -2;
  if (strcmp (category, "LC_MESSAGES") == 0)
    {
      struct stat64 st;

      if (stat64 (fname, &st) < 0)
	{
	  if (mkdir (fname, 0777) >= 0)
	    {
	      fd = -1;
	      errno = EISDIR;
	    }
	}
      else if (!S_ISREG (st.st_mode))
	{
	  fd = -1;
	  errno = EISDIR;
	}
    }

  /* Create the locale file with nlinks == 1; this avoids crashing processes
     which currently use the locale and damaging files belonging to other
     locales as well.  */
  if (fd == -2)
    {
      unlink (fname);
      fd = creat (fname, 0666);
    }

  if (fd == -1)
    {
      int save_err = errno;

      if (errno == EISDIR)
	{
	  sprintf (fname, "%1$s%2$s/SYS_%2$s", output_path, category);
	  unlink (fname);
	  fd = creat (fname, 0666);
	  if (fd == -1)
	    save_err = errno;
	}

      if (fd == -1)
	{
	  record_error (0, save_err, _("\
cannot open output file `%s' for category `%s'"), fname, category);
	  free (fname);
	  return;
	}
    }

#ifdef UIO_MAXIOV
  maxiov = UIO_MAXIOV;
#else
  maxiov = sysconf (_SC_UIO_MAXIOV);
#endif

  /* Write the data using writev.  But we must take care for the
     limitation of the implementation.  */
  for (cnt = 0; cnt < n_elem; cnt += step)
    {
      step = n_elem - cnt;
      if (maxiov > 0)
	step = MIN (maxiov, step);

      if (writev (fd, &vec[cnt], step) < 0)
	{
	  record_error (0, errno, _("\
failure while writing data for category `%s'"), category);
	  break;
	}
    }

  close (fd);

  /* Compare the file with the locale data files for the same category
     in other locales, and see if we can reuse it, to save disk space.
     If the user specified --no-hard-links to localedef then hard_links
     is false, other_paths remains NULL and we skip the optimization
     below.  The use of --no-hard-links is distribution specific since
     some distros have post-processing hard-link steps and so doing this
     here is a waste of time.  Worse than a waste of time in rpm-based
     distributions it can result in build determinism issues from
     build-to-build since some files may get a hard link in one pass but
     not in another (if the files happened to be created in parallel).  */
  if (hard_links)
    other_paths = siblings (output_path);

  /* If there are other paths, then walk the sibling paths looking for
     files with the same content so we can hard link and reduce disk
     space usage.  */
  if (other_paths != NULL)
    {
      struct stat64 fname_stat;

      if (lstat64 (fname, &fname_stat) >= 0
	  && S_ISREG (fname_stat.st_mode))
	{
	  const char *fname_tail = fname + strlen (output_path);
	  const char **other_p;
	  int seen_count;
	  ino_t *seen_inodes;

	  seen_count = 0;
	  for (other_p = other_paths; *other_p; other_p++)
	    seen_count++;
	  seen_inodes = (ino_t *) xmalloc (seen_count * sizeof (ino_t));
	  seen_count = 0;

	  for (other_p = other_paths; *other_p; other_p++)
	    {
	      const char *other_path = *other_p;
	      size_t other_path_len = strlen (other_path);
	      char *other_fname;
	      struct stat64 other_fname_stat;

	      other_fname =
		(char *) xmalloc (other_path_len + strlen (fname_tail) + 1);
	      memcpy (other_fname, other_path, other_path_len);
	      strcpy (other_fname + other_path_len, fname_tail);

	      if (lstat64 (other_fname, &other_fname_stat) >= 0
		  && S_ISREG (other_fname_stat.st_mode)
		  /* Consider only files on the same device.
		     Otherwise hard linking won't work anyway.  */
		  && other_fname_stat.st_dev == fname_stat.st_dev
		  /* Consider only files with the same permissions.
		     Otherwise there are security risks.  */
		  && other_fname_stat.st_uid == fname_stat.st_uid
		  && other_fname_stat.st_gid == fname_stat.st_gid
		  && other_fname_stat.st_mode == fname_stat.st_mode
		  /* Don't compare fname with itself.  */
		  && other_fname_stat.st_ino != fname_stat.st_ino
		  /* Files must have the same size, otherwise they
		     cannot be the same.  */
		  && other_fname_stat.st_size == fname_stat.st_size)
		{
		  /* Skip this file if we have already read it (under a
		     different name).  */
		  int i;

		  for (i = seen_count - 1; i >= 0; i--)
		    if (seen_inodes[i] == other_fname_stat.st_ino)
		      break;
		  if (i < 0)
		    {
		      /* Now compare fname and other_fname for real.  */
		      blksize_t blocksize;

#ifdef _STATBUF_ST_BLKSIZE
		      blocksize = MAX (fname_stat.st_blksize,
				       other_fname_stat.st_blksize);
		      if (blocksize > 8 * 1024)
			blocksize = 8 * 1024;
#else
		      blocksize = 8 * 1024;
#endif

		      if (compare_files (fname, other_fname,
					 fname_stat.st_size, blocksize) == 0)
			{
			  /* Found! other_fname is identical to fname.  */
			  /* Link other_fname to fname.  But use a temporary
			     file, in case hard links don't work on the
			     particular filesystem.  */
			  char * tmp_fname =
			    (char *) xmalloc (strlen (fname) + 4 + 1);

			  strcpy (stpcpy (tmp_fname, fname), ".tmp");

			  if (link (other_fname, tmp_fname) >= 0)
			    {
			      unlink (fname);
			      if (rename (tmp_fname, fname) < 0)
				{
				  record_error (0, errno, _("\
cannot create output file `%s' for category `%s'"), fname, category);
				}
			      free (tmp_fname);
			      free (other_fname);
			      break;
			    }
			  free (tmp_fname);
			}

		      /* Don't compare with this file a second time.  */
		      seen_inodes[seen_count++] = other_fname_stat.st_ino;
		    }
		}
	      free (other_fname);
	    }
	  free (seen_inodes);
	}
    }

  free (fname);
}


/* General handling of `copy'.  */
void
handle_copy (struct linereader *ldfile, const struct charmap_t *charmap,
	     const char *repertoire_name, struct localedef_t *result,
	     enum token_t token, int locale, const char *locale_name,
	     int ignore_content)
{
  struct token *now;
  int warned = 0;

  now = lr_token (ldfile, charmap, result, NULL, verbose);
  if (now->tok != tok_string)
    lr_error (ldfile, _("expecting string argument for `copy'"));
  else if (!ignore_content)
    {
      if (now->val.str.startmb == NULL)
	lr_error (ldfile, _("\
locale name should consist only of portable characters"));
      else
	{
	  (void) add_to_readlist (locale, now->val.str.startmb,
				  repertoire_name, 1, NULL);
	  result->copy_name[locale] = now->val.str.startmb;
	}
    }

  lr_ignore_rest (ldfile, now->tok == tok_string);

  /* The rest of the line must be empty and the next keyword must be
     `END xxx'.  */
  while ((now = lr_token (ldfile, charmap, result, NULL, verbose))->tok
	 != tok_end && now->tok != tok_eof)
    {
      if (warned == 0)
	{
	  lr_error (ldfile, _("\
no other keyword shall be specified when `copy' is used"));
	  warned = 1;
	}

      lr_ignore_rest (ldfile, 0);
    }

  if (now->tok != tok_eof)
    {
      /* Handle `END xxx'.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);

      if (now->tok != token)
	lr_error (ldfile, _("\
`%1$s' definition does not end with `END %1$s'"), locale_name);

      lr_ignore_rest (ldfile, now->tok == token);
    }
  else
    /* When we come here we reached the end of the file.  */
    lr_error (ldfile, _("%s: premature end of file"), locale_name);
}
