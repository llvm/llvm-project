/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.  */

#include <libioP.h>
#include <dlfcn.h>
#include <wchar.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <langinfo.h>
#include <locale/localeinfo.h>
#include <wcsmbs/wcsmbsload.h>
#include <iconv/gconv_int.h>
#include <shlib-compat.h>
#include <sysdep.h>


/* Return orientation of stream.  If mode is nonzero try to change
   the orientation first.  */
#undef _IO_fwide
int
_IO_fwide (FILE *fp, int mode)
{
  /* Normalize the value.  */
  mode = mode < 0 ? -1 : (mode == 0 ? 0 : 1);

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
  if (__glibc_unlikely (&_IO_stdin_used == NULL) && _IO_legacy_file (fp))
    /* This is for a stream in the glibc 2.0 format.  */
    return -1;
#endif

  /* The orientation already has been determined.  */
  if (fp->_mode != 0
      /* Or the caller simply wants to know about the current orientation.  */
      || mode == 0)
    return fp->_mode;

  /* Set the orientation appropriately.  */
  if (mode > 0)
    {
      struct _IO_codecvt *cc = fp->_codecvt = &fp->_wide_data->_codecvt;

      fp->_wide_data->_IO_read_ptr = fp->_wide_data->_IO_read_end;
      fp->_wide_data->_IO_write_ptr = fp->_wide_data->_IO_write_base;

      /* Get the character conversion functions based on the currently
	 selected locale for LC_CTYPE.  */
      {
	/* Clear the state.  We start all over again.  */
	memset (&fp->_wide_data->_IO_state, '\0', sizeof (__mbstate_t));
	memset (&fp->_wide_data->_IO_last_state, '\0', sizeof (__mbstate_t));

	struct gconv_fcts fcts;
	__wcsmbs_clone_conv (&fcts);
	assert (fcts.towc_nsteps == 1);
	assert (fcts.tomb_nsteps == 1);

	cc->__cd_in.step = fcts.towc;

	cc->__cd_in.step_data.__invocation_counter = 0;
	cc->__cd_in.step_data.__internal_use = 1;
	cc->__cd_in.step_data.__flags = __GCONV_IS_LAST;
	cc->__cd_in.step_data.__statep = &fp->_wide_data->_IO_state;

	cc->__cd_out.step = fcts.tomb;

	cc->__cd_out.step_data.__invocation_counter = 0;
	cc->__cd_out.step_data.__internal_use = 1;
	cc->__cd_out.step_data.__flags = __GCONV_IS_LAST | __GCONV_TRANSLIT;
	cc->__cd_out.step_data.__statep = &fp->_wide_data->_IO_state;
      }

      /* From now on use the wide character callback functions.  */
      _IO_JUMPS_FILE_plus (fp) = fp->_wide_data->_wide_vtable;
    }

  /* Set the mode now.  */
  fp->_mode = mode;

  return mode;
}


enum __codecvt_result
__libio_codecvt_out (struct _IO_codecvt *codecvt, __mbstate_t *statep,
		     const wchar_t *from_start, const wchar_t *from_end,
		     const wchar_t **from_stop, char *to_start, char *to_end,
		     char **to_stop)
{
  enum __codecvt_result result;

  struct __gconv_step *gs = codecvt->__cd_out.step;
  int status;
  size_t dummy;
  const unsigned char *from_start_copy = (unsigned char *) from_start;

  codecvt->__cd_out.step_data.__outbuf = (unsigned char *) to_start;
  codecvt->__cd_out.step_data.__outbufend = (unsigned char *) to_end;
  codecvt->__cd_out.step_data.__statep = statep;

  __gconv_fct fct = gs->__fct;
#ifdef PTR_DEMANGLE
  if (gs->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  status = DL_CALL_FCT (fct,
			(gs, &codecvt->__cd_out.step_data, &from_start_copy,
			 (const unsigned char *) from_end, NULL,
			 &dummy, 0, 0));

  *from_stop = (wchar_t *) from_start_copy;
  *to_stop = (char *) codecvt->__cd_out.step_data.__outbuf;

  switch (status)
    {
    case __GCONV_OK:
    case __GCONV_EMPTY_INPUT:
      result = __codecvt_ok;
      break;

    case __GCONV_FULL_OUTPUT:
    case __GCONV_INCOMPLETE_INPUT:
      result = __codecvt_partial;
      break;

    default:
      result = __codecvt_error;
      break;
    }

  return result;
}


enum __codecvt_result
__libio_codecvt_in (struct _IO_codecvt *codecvt, __mbstate_t *statep,
		    const char *from_start, const char *from_end,
		    const char **from_stop,
		    wchar_t *to_start, wchar_t *to_end, wchar_t **to_stop)
{
  enum __codecvt_result result;

  struct __gconv_step *gs = codecvt->__cd_in.step;
  int status;
  size_t dummy;
  const unsigned char *from_start_copy = (unsigned char *) from_start;

  codecvt->__cd_in.step_data.__outbuf = (unsigned char *) to_start;
  codecvt->__cd_in.step_data.__outbufend = (unsigned char *) to_end;
  codecvt->__cd_in.step_data.__statep = statep;

  __gconv_fct fct = gs->__fct;
#ifdef PTR_DEMANGLE
  if (gs->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  status = DL_CALL_FCT (fct,
			(gs, &codecvt->__cd_in.step_data, &from_start_copy,
			 (const unsigned char *) from_end, NULL,
			 &dummy, 0, 0));

  *from_stop = (const char *) from_start_copy;
  *to_stop = (wchar_t *) codecvt->__cd_in.step_data.__outbuf;

  switch (status)
    {
    case __GCONV_OK:
    case __GCONV_EMPTY_INPUT:
      result = __codecvt_ok;
      break;

    case __GCONV_FULL_OUTPUT:
    case __GCONV_INCOMPLETE_INPUT:
      result = __codecvt_partial;
      break;

    default:
      result = __codecvt_error;
      break;
    }

  return result;
}


int
__libio_codecvt_encoding (struct _IO_codecvt *codecvt)
{
  /* See whether the encoding is stateful.  */
  if (codecvt->__cd_in.step->__stateful)
    return -1;
  /* Fortunately not.  Now determine the input bytes for the conversion
     necessary for each wide character.  */
  if (codecvt->__cd_in.step->__min_needed_from
      != codecvt->__cd_in.step->__max_needed_from)
    /* Not a constant value.  */
    return 0;

  return codecvt->__cd_in.step->__min_needed_from;
}


int
__libio_codecvt_length (struct _IO_codecvt *codecvt, __mbstate_t *statep,
			const char *from_start, const char *from_end,
			size_t max)
{
  int result;
  const unsigned char *cp = (const unsigned char *) from_start;
  wchar_t to_buf[max];
  struct __gconv_step *gs = codecvt->__cd_in.step;
  size_t dummy;

  codecvt->__cd_in.step_data.__outbuf = (unsigned char *) to_buf;
  codecvt->__cd_in.step_data.__outbufend = (unsigned char *) &to_buf[max];
  codecvt->__cd_in.step_data.__statep = statep;

  __gconv_fct fct = gs->__fct;
#ifdef PTR_DEMANGLE
  if (gs->__shlib_handle != NULL)
    PTR_DEMANGLE (fct);
#endif

  DL_CALL_FCT (fct,
	       (gs, &codecvt->__cd_in.step_data, &cp,
		(const unsigned char *) from_end, NULL,
		&dummy, 0, 0));

  result = cp - (const unsigned char *) from_start;

  return result;
}
