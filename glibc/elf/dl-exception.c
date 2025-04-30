/* ld.so error exception allocation and deallocation.
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

#include <ldsodefs.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <_itoa.h>

/* This message we return as a last resort.  We define the string in a
   variable since we have to avoid freeing it and so have to enable
   a pointer comparison.  See below and in dlfcn/dlerror.c.  */
static const char _dl_out_of_memory[] = "out of memory";

/* Call free in the main libc.so.  This allows other namespaces to
   free pointers on the main libc heap, via GLRO (dl_error_free).  It
   also avoids calling free on the special, pre-allocated
   out-of-memory error message.  */
void
_dl_error_free (void *ptr)
{
  if (ptr != _dl_out_of_memory)
    free (ptr);
}

/* Dummy allocation object used if allocating the message buffer
   fails.  */
static void
oom_exception (struct dl_exception *exception)
{
  exception->objname = "";
  exception->errstring = _dl_out_of_memory;
  exception->message_buffer = NULL;
}

static void
__attribute__ ((noreturn))
length_mismatch (void)
{
  _dl_fatal_printf ("Fatal error: "
                    "length accounting in _dl_exception_create_format\n");
}

/* Adjust the message buffer to indicate whether it is possible to
   free it.  EXCEPTION->errstring must be a potentially deallocatable
   pointer.  */
static void
adjust_message_buffer (struct dl_exception *exception)
{
  /* If the main executable is relocated it means the libc's malloc
     is used.  */
  bool malloced = true;
#ifdef SHARED
  malloced = (GL(dl_ns)[LM_ID_BASE]._ns_loaded != NULL
              && (GL(dl_ns)[LM_ID_BASE]._ns_loaded->l_relocated != 0));
#endif
  if (malloced)
    exception->message_buffer = (char *) exception->errstring;
  else
    exception->message_buffer = NULL;
}

void
_dl_exception_create (struct dl_exception *exception, const char *objname,
                      const char *errstring)
{
  if (objname == NULL)
    objname = "";
  size_t len_objname = strlen (objname) + 1;
  size_t len_errstring = strlen (errstring) + 1;
  char *errstring_copy = malloc (len_objname + len_errstring);
  if (errstring_copy != NULL)
    {
      /* Make a copy of the object file name and the error string.  */
      exception->objname = memcpy (__mempcpy (errstring_copy,
                                              errstring, len_errstring),
                                   objname, len_objname);
      exception->errstring = errstring_copy;
      adjust_message_buffer (exception);
    }
  else
    oom_exception (exception);
}
rtld_hidden_def (_dl_exception_create)

void
_dl_exception_create_format (struct dl_exception *exception, const char *objname,
                             const char *fmt, ...)
{
  if (objname == NULL)
    objname = "";
  size_t len_objname = strlen (objname) + 1;
  /* Compute the length of the result.  Include room for two NUL
     bytes.  */
  size_t length = len_objname + 1;
  {
    va_list ap;
    va_start (ap, fmt);
    for (const char *p = fmt; *p != '\0'; ++p)
      if (*p == '%')
        {
          ++p;
          switch (*p)
            {
            case 's':
              length += strlen (va_arg (ap, const char *));
              break;
	      /* Recognize the l modifier.  It is only important on some
		 platforms where long and int have a different size.  We
		 can use the same code for size_t.  */
	    case 'l':
	    case 'z':
	      if (p[1] == 'x')
		{
		  length += LONG_WIDTH / 4;
		  ++p;
		  break;
		}
	      /* Fall through.  */
	    case 'x':
	      length += INT_WIDTH / 4;
	      break;
            default:
              /* Assumed to be '%'.  */
              ++length;
              break;
            }
        }
      else
        ++length;
    va_end (ap);
  }

  if (length > PTRDIFF_MAX)
    {
      oom_exception (exception);
      return;
    }
  char *errstring = malloc (length);
  if (errstring == NULL)
    {
      oom_exception (exception);
      return;
    }
  exception->errstring = errstring;
  adjust_message_buffer (exception);

  /* Copy the error message to errstring.  */
  {
    /* Next byte to be written in errstring.  */
    char *wptr = errstring;
    /* End of the allocated string.  */
    char *const end = errstring + length;

    va_list ap;
    va_start (ap, fmt);

    for (const char *p = fmt; *p != '\0'; ++p)
      if (*p == '%')
        {
          ++p;
          switch (*p)
            {
            case 's':
              {
                const char *ptr = va_arg (ap, const char *);
                size_t len_ptr = strlen (ptr);
                if (len_ptr > end - wptr)
                  length_mismatch ();
                wptr = __mempcpy (wptr, ptr, len_ptr);
              }
              break;
            case '%':
              if (wptr == end)
                length_mismatch ();
              *wptr = '%';
              ++wptr;
              break;
	    case 'x':
	      {
		unsigned long int num = va_arg (ap, unsigned int);
		char *start = wptr;
		wptr += INT_WIDTH / 4;
		char *cp = _itoa (num, wptr, 16, 0);
		/* Pad to the full width with 0.  */
		while (cp != start)
		  *--cp = '0';
	      }
	      break;
	    case 'l':
	    case 'z':
	      if (p[1] == 'x')
		{
		  unsigned long int num = va_arg (ap, unsigned long int);
		  char *start = wptr;
		  wptr += LONG_WIDTH / 4;
		  char *cp = _itoa (num, wptr, 16, 0);
		  /* Pad to the full width with 0.  */
		  while (cp != start)
		    *--cp = '0';
		  ++p;
		  break;
		}
	       /* FALLTHROUGH */
            default:
              _dl_fatal_printf ("Fatal error:"
                                " invalid format in exception string\n");
            }
        }
      else
        {
          if (wptr == end)
            length_mismatch ();
          *wptr = *p;
          ++wptr;
        }

    if (wptr == end)
      length_mismatch ();
    *wptr = '\0';
    ++wptr;
    if (len_objname != end - wptr)
      length_mismatch ();
    exception->objname = memcpy (wptr, objname, len_objname);
    va_end (ap);
  }
}
rtld_hidden_def (_dl_exception_create_format)

void
_dl_exception_free (struct dl_exception *exception)
{
  free (exception->message_buffer);
  exception->objname = NULL;
  exception->errstring = NULL;
  exception->message_buffer = NULL;
}
rtld_hidden_def (_dl_exception_free)
