/* Miscellaneous support functions for dynamic linker
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <assert.h>
#include <fcntl.h>
#include <ldsodefs.h>
#include <limits.h>
#include <link.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sysdep.h>
#include <_itoa.h>
#include <dl-writev.h>
#include <not-cancel.h>

/* Read the whole contents of FILE into new mmap'd space with given
   protections.  *SIZEP gets the size of the file.  On error MAP_FAILED
   is returned.  */

void *
_dl_sysdep_read_whole_file (const char *file, size_t *sizep, int prot)
{
  void *result = MAP_FAILED;
  struct __stat64_t64 st;
  int fd = __open64_nocancel (file, O_RDONLY | O_CLOEXEC);
  if (fd >= 0)
    {
      if (__fstat64_time64 (fd, &st) >= 0)
	{
	  *sizep = st.st_size;

	  /* No need to map the file if it is empty.  */
	  if (*sizep != 0)
	    /* Map a copy of the file contents.  */
	    result = __mmap (NULL, *sizep, prot,
#ifdef MAP_COPY
			     MAP_COPY
#else
			     MAP_PRIVATE
#endif
#ifdef MAP_FILE
			     | MAP_FILE
#endif
			     , fd, 0);
	}
      __close_nocancel (fd);
    }
  return result;
}


/* Bare-bones printf implementation.  This function only knows about
   the formats and flags needed and can handle only up to 64 stripes in
   the output.  */
static void
_dl_debug_vdprintf (int fd, int tag_p, const char *fmt, va_list arg)
{
# define NIOVMAX 64
  struct iovec iov[NIOVMAX];
  int niov = 0;
  pid_t pid = 0;
  char pidbuf[12];

  while (*fmt != '\0')
    {
      const char *startp = fmt;

      if (tag_p > 0)
	{
	  /* Generate the tag line once.  It consists of the PID and a
	     colon followed by a tab.  */
	  if (pid == 0)
	    {
	      char *p;
	      pid = __getpid ();
	      assert (pid >= 0 && sizeof (pid_t) <= 4);
	      p = _itoa (pid, &pidbuf[10], 10, 0);
	      while (p > pidbuf)
		*--p = ' ';
	      pidbuf[10] = ':';
	      pidbuf[11] = '\t';
	    }

	  /* Append to the output.  */
	  assert (niov < NIOVMAX);
	  iov[niov].iov_len = 12;
	  iov[niov++].iov_base = pidbuf;

	  /* No more tags until we see the next newline.  */
	  tag_p = -1;
	}

      /* Skip everything except % and \n (if tags are needed).  */
      while (*fmt != '\0' && *fmt != '%' && (! tag_p || *fmt != '\n'))
	++fmt;

      /* Append constant string.  */
      assert (niov < NIOVMAX);
      if ((iov[niov].iov_len = fmt - startp) != 0)
	iov[niov++].iov_base = (char *) startp;

      if (*fmt == '%')
	{
	  /* It is a format specifier.  */
	  char fill = ' ';
	  int width = -1;
	  int prec = -1;
#if LONG_MAX != INT_MAX
	  int long_mod = 0;
#endif

	  /* Recognize zero-digit fill flag.  */
	  if (*++fmt == '0')
	    {
	      fill = '0';
	      ++fmt;
	    }

	  /* See whether with comes from a parameter.  Note that no other
	     way to specify the width is implemented.  */
	  if (*fmt == '*')
	    {
	      width = va_arg (arg, int);
	      ++fmt;
	    }

	  /* Handle precision.  */
	  if (*fmt == '.' && fmt[1] == '*')
	    {
	      prec = va_arg (arg, int);
	      fmt += 2;
	    }

	  /* Recognize the l modifier.  It is only important on some
	     platforms where long and int have a different size.  We
	     can use the same code for size_t.  */
	  if (*fmt == 'l' || *fmt == 'Z' || *fmt == 'z')
	    {
#if LONG_MAX != INT_MAX
	      long_mod = 1;
#endif
	      ++fmt;
	    }

	  switch (*fmt)
	    {
	      /* Integer formatting.  */
	    case 'd':
	    case 'u':
	    case 'x':
	      {
		/* We have to make a difference if long and int have a
		   different size.  */
#if LONG_MAX != INT_MAX
		unsigned long int num = (long_mod
					 ? va_arg (arg, unsigned long int)
					 : va_arg (arg, unsigned int));
#else
		unsigned long int num = va_arg (arg, unsigned int);
#endif
		bool negative = false;
		if (*fmt == 'd')
		  {
#if LONG_MAX != INT_MAX
		    if (long_mod)
		      {
			if ((long int) num < 0)
			  negative = true;
		      }
		    else
		      {
			if ((int) num < 0)
			  {
			    num = (unsigned int) num;
			    negative = true;
			  }
		      }
#else
		    if ((int) num < 0)
		      negative = true;
#endif
		  }

		/* We use alloca() to allocate the buffer with the most
		   pessimistic guess for the size.  Using alloca() allows
		   having more than one integer formatting in a call.  */
		char *buf = (char *) alloca (1 + 3 * sizeof (unsigned long int));
		char *endp = &buf[1 + 3 * sizeof (unsigned long int)];
		char *cp = _itoa (num, endp, *fmt == 'x' ? 16 : 10, 0);

		/* Pad to the width the user specified.  */
		if (width != -1)
		  while (endp - cp < width)
		    *--cp = fill;

		if (negative)
		  *--cp = '-';

		iov[niov].iov_base = cp;
		iov[niov].iov_len = endp - cp;
		++niov;
	      }
	      break;

	    case 's':
	      /* Get the string argument.  */
	      iov[niov].iov_base = va_arg (arg, char *);
	      iov[niov].iov_len = strlen (iov[niov].iov_base);
	      if (prec != -1)
		iov[niov].iov_len = MIN ((size_t) prec, iov[niov].iov_len);
	      ++niov;
	      break;

	    case '%':
	      iov[niov].iov_base = (void *) fmt;
	      iov[niov].iov_len = 1;
	      ++niov;
	      break;

	    default:
	      assert (! "invalid format specifier");
	    }
	  ++fmt;
	}
      else if (*fmt == '\n')
	{
	  /* See whether we have to print a single newline character.  */
	  if (fmt == startp)
	    {
	      iov[niov].iov_base = (char *) startp;
	      iov[niov++].iov_len = 1;
	    }
	  else
	    /* No, just add it to the rest of the string.  */
	    ++iov[niov - 1].iov_len;

	  /* Next line, print a tag again.  */
	  tag_p = 1;
	  ++fmt;
	}
    }

  /* Finally write the result.  */
  _dl_writev (fd, iov, niov);
}


/* Write to debug file.  */
void
_dl_debug_printf (const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (GLRO(dl_debug_fd), 1, fmt, arg);
  va_end (arg);
}


/* Write to debug file but don't start with a tag.  */
void
_dl_debug_printf_c (const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (GLRO(dl_debug_fd), -1, fmt, arg);
  va_end (arg);
}


/* Write the given file descriptor.  */
void
_dl_dprintf (int fd, const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (fd, 0, fmt, arg);
  va_end (arg);
}

void
_dl_printf (const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (STDOUT_FILENO, 0, fmt, arg);
  va_end (arg);
}

void
_dl_error_printf (const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (STDERR_FILENO, 0, fmt, arg);
  va_end (arg);
}

void
_dl_fatal_printf (const char *fmt, ...)
{
  va_list arg;

  va_start (arg, fmt);
  _dl_debug_vdprintf (STDERR_FILENO, 0, fmt, arg);
  va_end (arg);
  _exit (127);
}
rtld_hidden_def (_dl_fatal_printf)

/* Test whether given NAME matches any of the names of the given object.  */
int
_dl_name_match_p (const char *name, const struct link_map *map)
{
  if (strcmp (name, map->l_name) == 0)
    return 1;

  struct libname_list *runp = map->l_libname;

  while (runp != NULL)
    if (strcmp (name, runp->name) == 0)
      return 1;
    else
      /* Synchronize with the release MO store in add_name_to_object.
	 See CONCURRENCY NOTES in add_name_to_object in dl-load.c.  */
      runp = atomic_load_acquire (&runp->next);

  return 0;
}


unsigned long int
_dl_higher_prime_number (unsigned long int n)
{
  /* These are primes that are near, but slightly smaller than, a
     power of two.  */
  static const uint32_t primes[] = {
    UINT32_C (7),
    UINT32_C (13),
    UINT32_C (31),
    UINT32_C (61),
    UINT32_C (127),
    UINT32_C (251),
    UINT32_C (509),
    UINT32_C (1021),
    UINT32_C (2039),
    UINT32_C (4093),
    UINT32_C (8191),
    UINT32_C (16381),
    UINT32_C (32749),
    UINT32_C (65521),
    UINT32_C (131071),
    UINT32_C (262139),
    UINT32_C (524287),
    UINT32_C (1048573),
    UINT32_C (2097143),
    UINT32_C (4194301),
    UINT32_C (8388593),
    UINT32_C (16777213),
    UINT32_C (33554393),
    UINT32_C (67108859),
    UINT32_C (134217689),
    UINT32_C (268435399),
    UINT32_C (536870909),
    UINT32_C (1073741789),
    UINT32_C (2147483647),
				       /* 4294967291L */
    UINT32_C (2147483647) + UINT32_C (2147483644)
  };

  const uint32_t *low = &primes[0];
  const uint32_t *high = &primes[sizeof (primes) / sizeof (primes[0])];

  while (low != high)
    {
      const uint32_t *mid = low + (high - low) / 2;
      if (n > *mid)
       low = mid + 1;
      else
       high = mid;
    }

#if 0
  /* If we've run out of primes, abort.  */
  if (n > *low)
    {
      fprintf (stderr, "Cannot find prime bigger than %lu\n", n);
      abort ();
    }
#endif

  return *low;
}

/* A stripped down strtoul-like implementation for very early use.  It
   does not set errno if the result is outside bounds because it may get
   called before errno may have been set up.  */

uint64_t
_dl_strtoul (const char *nptr, char **endptr)
{
  uint64_t result = 0;
  bool positive = true;
  unsigned max_digit;

  while (*nptr == ' ' || *nptr == '\t')
    ++nptr;

  if (*nptr == '-')
    {
      positive = false;
      ++nptr;
    }
  else if (*nptr == '+')
    ++nptr;

  if (*nptr < '0' || *nptr > '9')
    {
      if (endptr != NULL)
	*endptr = (char *) nptr;
      return 0UL;
    }

  int base = 10;
  max_digit = 9;
  if (*nptr == '0')
    {
      if (nptr[1] == 'x' || nptr[1] == 'X')
	{
	  base = 16;
	  nptr += 2;
	}
      else
	{
	  base = 8;
	  max_digit = 7;
	}
    }

  while (1)
    {
      int digval;
      if (*nptr >= '0' && *nptr <= '0' + max_digit)
        digval = *nptr - '0';
      else if (base == 16)
        {
	  if (*nptr >= 'a' && *nptr <= 'f')
	    digval = *nptr - 'a' + 10;
	  else if (*nptr >= 'A' && *nptr <= 'F')
	    digval = *nptr - 'A' + 10;
	  else
	    break;
	}
      else
        break;

      if (result >= (UINT64_MAX - digval) / base)
	{
	  if (endptr != NULL)
	    *endptr = (char *) nptr;
	  return UINT64_MAX;
	}
      result *= base;
      result += digval;
      ++nptr;
    }

  if (endptr != NULL)
    *endptr = (char *) nptr;

  /* Avoid 64-bit multiplication.  */
  if (!positive)
    result = -result;

  return result;
}
