/* Template for error handling for runtime dynamic linker.
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

/* The following macro needs to be defined before including this
   skeleton file:

   DL_ERROR_BOOTSTRAP

     If 1, do not use TLS and implement _dl_signal_cerror and
     _dl_receive_error.  If 0, TLS is used, and the variants with
     error callbacks are not provided.  */


#include <libintl.h>
#include <setjmp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ldsodefs.h>
#include <stdio.h>

/* This structure communicates state between _dl_catch_error and
   _dl_signal_error.  */
struct catch
  {
    struct dl_exception *exception; /* The exception data is stored there.  */
    volatile int *errcode;	/* Return value of _dl_signal_error.  */
    jmp_buf env;		/* longjmp here on error.  */
  };

/* Multiple threads at once can use the `_dl_catch_error' function.  The
   calls can come from `_dl_map_object_deps', `_dlerror_run', or from
   any of the libc functionality which loads dynamic objects (NSS, iconv).
   Therefore we have to be prepared to save the state in thread-local
   memory.  */
#if !DL_ERROR_BOOTSTRAP
static __thread struct catch *catch_hook attribute_tls_model_ie;
#else
/* The version of this code in ld.so cannot use thread-local variables
   and is used during bootstrap only.  */
static struct catch *catch_hook;
#endif

#if DL_ERROR_BOOTSTRAP
/* This points to a function which is called when an continuable error is
   received.  Unlike the handling of `catch' this function may return.
   The arguments will be the `errstring' and `objname'.

   Since this functionality is not used in normal programs (only in ld.so)
   we do not care about multi-threaded programs here.  We keep this as a
   global variable.  */
static receiver_fct receiver;
#endif /* DL_ERROR_BOOTSTRAP */

/* Lossage while resolving the program's own symbols is always fatal.  */
static void
__attribute__ ((noreturn))
fatal_error (int errcode, const char *objname, const char *occasion,
	     const char *errstring)
{
  char buffer[1024];
  _dl_fatal_printf ("%s: %s: %s%s%s%s%s\n",
		    RTLD_PROGNAME,
		    occasion ?: N_("error while loading shared libraries"),
		    objname, *objname ? ": " : "",
		    errstring, errcode ? ": " : "",
		    (errcode
		     ? __strerror_r (errcode, buffer, sizeof buffer)
		     : ""));
}

void
_dl_signal_exception (int errcode, struct dl_exception *exception,
		      const char *occasion)
{
  struct catch *lcatch = catch_hook;
  if (lcatch != NULL)
    {
      *lcatch->exception = *exception;
      *lcatch->errcode = errcode;

      /* We do not restore the signal mask because none was saved.  */
      __longjmp (lcatch->env[0].__jmpbuf, 1);
    }
  else
    fatal_error (errcode, exception->objname, occasion, exception->errstring);
}
libc_hidden_def (_dl_signal_exception)

void
_dl_signal_error (int errcode, const char *objname, const char *occation,
		  const char *errstring)
{
  struct catch *lcatch = catch_hook;

  if (! errstring)
    errstring = N_("DYNAMIC LINKER BUG!!!");

  if (lcatch != NULL)
    {
      _dl_exception_create (lcatch->exception, objname, errstring);
      *lcatch->errcode = errcode;

      /* We do not restore the signal mask because none was saved.  */
      __longjmp (lcatch->env[0].__jmpbuf, 1);
    }
  else
    fatal_error (errcode, objname, occation, errstring);
}
libc_hidden_def (_dl_signal_error)


#if DL_ERROR_BOOTSTRAP
void
_dl_signal_cexception (int errcode, struct dl_exception *exception,
		       const char *occasion)
{
  if (__builtin_expect (GLRO(dl_debug_mask)
			& ~(DL_DEBUG_STATISTICS|DL_DEBUG_PRELINK), 0))
    _dl_debug_printf ("%s: error: %s: %s (%s)\n",
		      exception->objname, occasion,
		      exception->errstring, receiver ? "continued" : "fatal");

  if (receiver)
    {
      /* We are inside _dl_receive_error.  Call the user supplied
	 handler and resume the work.  The receiver will still be
	 installed.  */
      (*receiver) (errcode, exception->objname, exception->errstring);
    }
  else
    _dl_signal_exception (errcode, exception, occasion);
}

void
_dl_signal_cerror (int errcode, const char *objname, const char *occation,
		   const char *errstring)
{
  if (__builtin_expect (GLRO(dl_debug_mask)
			& ~(DL_DEBUG_STATISTICS|DL_DEBUG_PRELINK), 0))
    _dl_debug_printf ("%s: error: %s: %s (%s)\n", objname, occation,
		      errstring, receiver ? "continued" : "fatal");

  if (receiver)
    {
      /* We are inside _dl_receive_error.  Call the user supplied
	 handler and resume the work.  The receiver will still be
	 installed.  */
      (*receiver) (errcode, objname, errstring);
    }
  else
    _dl_signal_error (errcode, objname, occation, errstring);
}
#endif /* DL_ERROR_BOOTSTRAP */

int
_dl_catch_exception (struct dl_exception *exception,
		     void (*operate) (void *), void *args)
{
  /* If exception is NULL, temporarily disable exception handling.
     Exceptions during operate (args) are fatal.  */
  if (exception == NULL)
    {
      struct catch *const old = catch_hook;
      catch_hook = NULL;
      operate (args);
      /* If we get here, the operation was successful.  */
      catch_hook = old;
      return 0;
    }

  /* We need not handle `receiver' since setting a `catch' is handled
     before it.  */

  /* Only this needs to be marked volatile, because it is the only local
     variable that gets changed between the setjmp invocation and the
     longjmp call.  All others are just set here (before setjmp) and read
     in _dl_signal_error (before longjmp).  */
  volatile int errcode;

  struct catch c;
  /* Don't use an initializer since we don't need to clear C.env.  */
  c.exception = exception;
  c.errcode = &errcode;

  struct catch *const old = catch_hook;
  catch_hook = &c;

  /* Do not save the signal mask.  */
  if (__builtin_expect (__sigsetjmp (c.env, 0), 0) == 0)
    {
      (*operate) (args);
      catch_hook = old;
      *exception = (struct dl_exception) { NULL };
      return 0;
    }

  /* We get here only if we longjmp'd out of OPERATE.
     _dl_signal_exception has already stored values into
     *EXCEPTION.  */
  catch_hook = old;
  return errcode;
}
libc_hidden_def (_dl_catch_exception)

int
_dl_catch_error (const char **objname, const char **errstring,
		 bool *mallocedp, void (*operate) (void *), void *args)
{
  struct dl_exception exception;
  int errorcode = _dl_catch_exception (&exception, operate, args);
  *objname = exception.objname;
  *errstring = exception.errstring;
  *mallocedp = exception.message_buffer == exception.errstring;
  return errorcode;
}
libc_hidden_def (_dl_catch_error)

#if DL_ERROR_BOOTSTRAP
void
_dl_receive_error (receiver_fct fct, void (*operate) (void *), void *args)
{
  struct catch *old_catch = catch_hook;
  receiver_fct old_receiver = receiver;

  /* Set the new values.  */
  catch_hook = NULL;
  receiver = fct;

  (*operate) (args);

  catch_hook = old_catch;
  receiver = old_receiver;
}

/* Forwarder used for initializing GLRO (_dl_catch_error).  */
int
_rtld_catch_error (const char **objname, const char **errstring,
		   bool *mallocedp, void (*operate) (void *),
		   void *args)
{
  /* The reference to _dl_catch_error will eventually be relocated to
     point to the implementation in libc.so.  */
  return _dl_catch_error (objname, errstring, mallocedp, operate, args);
}

#endif /* DL_ERROR_BOOTSTRAP */
