/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
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

/* NOTE: libio is now exclusively used only by glibc since libstdc++ has its
   own implementation.  As a result, functions that were implemented for C++
   (like *sputn) may no longer have C++ semantics.  This is of course only
   relevant for internal callers of these functions since these functions are
   not intended for external use otherwise.

   FIXME: All of the C++ cruft eventually needs to go away.  */

#ifndef _LIBIOP_H
#define _LIBIOP_H 1

#include <stddef.h>

#include <errno.h>
#include <libc-lock.h>

#include <math_ldbl_opt.h>

#include <stdio.h>
#include <libio/libio.h>
#include "iolibio.h"

#include <shlib-compat.h>

/* For historical reasons this is the name of the sysdeps header that
   adjusts the libio configuration.  */
#include <_G_config.h>

#define _IO_seek_set 0
#define _IO_seek_cur 1
#define _IO_seek_end 2

/* THE JUMPTABLE FUNCTIONS.

 * The _IO_FILE type is used to implement the FILE type in GNU libc,
 * as well as the streambuf class in GNU iostreams for C++.
 * These are all the same, just used differently.
 * An _IO_FILE (or FILE) object is allows followed by a pointer to
 * a jump table (of pointers to functions).  The pointer is accessed
 * with the _IO_JUMPS macro.  The jump table has an eccentric format,
 * so as to be compatible with the layout of a C++ virtual function table.
 * (as implemented by g++).  When a pointer to a streambuf object is
 * coerced to an (FILE*), then _IO_JUMPS on the result just
 * happens to point to the virtual function table of the streambuf.
 * Thus the _IO_JUMPS function table used for C stdio/libio does
 * double duty as the virtual function table for C++ streambuf.
 *
 * The entries in the _IO_JUMPS function table (and hence also the
 * virtual functions of a streambuf) are described below.
 * The first parameter of each function entry is the _IO_FILE/streambuf
 * object being acted on (i.e. the 'this' parameter).
 */

/* Setting this macro to 1 enables the use of the _vtable_offset bias
   in _IO_JUMPS_FUNCS, below.  This is only needed for new-format
   _IO_FILE in libc that must support old binaries (see oldfileops.c).  */
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1) && !defined _IO_USE_OLD_IO_FILE
# define _IO_JUMPS_OFFSET 1
#else
# define _IO_JUMPS_OFFSET 0
#endif

/* Type of MEMBER in struct type TYPE.  */
#define _IO_MEMBER_TYPE(TYPE, MEMBER) __typeof__ (((TYPE){}).MEMBER)

/* Essentially ((TYPE *) THIS)->MEMBER, but avoiding the aliasing
   violation in case THIS has a different pointer type.  */
#define _IO_CAST_FIELD_ACCESS(THIS, TYPE, MEMBER) \
  (*(_IO_MEMBER_TYPE (TYPE, MEMBER) *)(((char *) (THIS)) \
				       + offsetof(TYPE, MEMBER)))

#define _IO_JUMPS(THIS) (THIS)->vtable
#define _IO_JUMPS_FILE_plus(THIS) \
  _IO_CAST_FIELD_ACCESS ((THIS), struct _IO_FILE_plus, vtable)
#define _IO_WIDE_JUMPS(THIS) \
  _IO_CAST_FIELD_ACCESS ((THIS), struct _IO_FILE, _wide_data)->_wide_vtable
#define _IO_CHECK_WIDE(THIS) \
  (_IO_CAST_FIELD_ACCESS ((THIS), struct _IO_FILE, _wide_data) != NULL)

#if _IO_JUMPS_OFFSET
# define _IO_JUMPS_FUNC(THIS) \
  (IO_validate_vtable                                                   \
   (*(struct _IO_jump_t **) ((void *) &_IO_JUMPS_FILE_plus (THIS)	\
			     + (THIS)->_vtable_offset)))
# define _IO_JUMPS_FUNC_UPDATE(THIS, VTABLE)				\
  (*(const struct _IO_jump_t **) ((void *) &_IO_JUMPS_FILE_plus (THIS)	\
				  + (THIS)->_vtable_offset) = (VTABLE))
# define _IO_vtable_offset(THIS) (THIS)->_vtable_offset
#else
# define _IO_JUMPS_FUNC(THIS) (IO_validate_vtable (_IO_JUMPS_FILE_plus (THIS)))
# define _IO_JUMPS_FUNC_UPDATE(THIS, VTABLE) \
  (_IO_JUMPS_FILE_plus (THIS) = (VTABLE))
# define _IO_vtable_offset(THIS) 0
#endif
#define _IO_WIDE_JUMPS_FUNC(THIS) _IO_WIDE_JUMPS(THIS)
#define JUMP_FIELD(TYPE, NAME) TYPE NAME
#define JUMP0(FUNC, THIS) (_IO_JUMPS_FUNC(THIS)->FUNC) (THIS)
#define JUMP1(FUNC, THIS, X1) (_IO_JUMPS_FUNC(THIS)->FUNC) (THIS, X1)
#define JUMP2(FUNC, THIS, X1, X2) (_IO_JUMPS_FUNC(THIS)->FUNC) (THIS, X1, X2)
#define JUMP3(FUNC, THIS, X1,X2,X3) (_IO_JUMPS_FUNC(THIS)->FUNC) (THIS, X1,X2, X3)
#define JUMP_INIT(NAME, VALUE) VALUE
#define JUMP_INIT_DUMMY JUMP_INIT(dummy, 0), JUMP_INIT (dummy2, 0)

#define WJUMP0(FUNC, THIS) (_IO_WIDE_JUMPS_FUNC(THIS)->FUNC) (THIS)
#define WJUMP1(FUNC, THIS, X1) (_IO_WIDE_JUMPS_FUNC(THIS)->FUNC) (THIS, X1)
#define WJUMP2(FUNC, THIS, X1, X2) (_IO_WIDE_JUMPS_FUNC(THIS)->FUNC) (THIS, X1, X2)
#define WJUMP3(FUNC, THIS, X1,X2,X3) (_IO_WIDE_JUMPS_FUNC(THIS)->FUNC) (THIS, X1,X2, X3)

/* The 'finish' function does any final cleaning up of an _IO_FILE object.
   It does not delete (free) it, but does everything else to finalize it.
   It matches the streambuf::~streambuf virtual destructor.  */
typedef void (*_IO_finish_t) (FILE *, int); /* finalize */
#define _IO_FINISH(FP) JUMP1 (__finish, FP, 0)
#define _IO_WFINISH(FP) WJUMP1 (__finish, FP, 0)

/* The 'overflow' hook flushes the buffer.
   The second argument is a character, or EOF.
   It matches the streambuf::overflow virtual function. */
typedef int (*_IO_overflow_t) (FILE *, int);
#define _IO_OVERFLOW(FP, CH) JUMP1 (__overflow, FP, CH)
#define _IO_WOVERFLOW(FP, CH) WJUMP1 (__overflow, FP, CH)

/* The 'underflow' hook tries to fills the get buffer.
   It returns the next character (as an unsigned char) or EOF.  The next
   character remains in the get buffer, and the get position is not changed.
   It matches the streambuf::underflow virtual function. */
typedef int (*_IO_underflow_t) (FILE *);
#define _IO_UNDERFLOW(FP) JUMP0 (__underflow, FP)
#define _IO_WUNDERFLOW(FP) WJUMP0 (__underflow, FP)

/* The 'uflow' hook returns the next character in the input stream
   (cast to unsigned char), and increments the read position;
   EOF is returned on failure.
   It matches the streambuf::uflow virtual function, which is not in the
   cfront implementation, but was added to C++ by the ANSI/ISO committee. */
#define _IO_UFLOW(FP) JUMP0 (__uflow, FP)
#define _IO_WUFLOW(FP) WJUMP0 (__uflow, FP)

/* The 'pbackfail' hook handles backing up.
   It matches the streambuf::pbackfail virtual function. */
typedef int (*_IO_pbackfail_t) (FILE *, int);
#define _IO_PBACKFAIL(FP, CH) JUMP1 (__pbackfail, FP, CH)
#define _IO_WPBACKFAIL(FP, CH) WJUMP1 (__pbackfail, FP, CH)

/* The 'xsputn' hook writes upto N characters from buffer DATA.
   Returns EOF or the number of character actually written.
   It matches the streambuf::xsputn virtual function. */
typedef size_t (*_IO_xsputn_t) (FILE *FP, const void *DATA,
				    size_t N);
#define _IO_XSPUTN(FP, DATA, N) JUMP2 (__xsputn, FP, DATA, N)
#define _IO_WXSPUTN(FP, DATA, N) WJUMP2 (__xsputn, FP, DATA, N)

/* The 'xsgetn' hook reads upto N characters into buffer DATA.
   Returns the number of character actually read.
   It matches the streambuf::xsgetn virtual function. */
typedef size_t (*_IO_xsgetn_t) (FILE *FP, void *DATA, size_t N);
#define _IO_XSGETN(FP, DATA, N) JUMP2 (__xsgetn, FP, DATA, N)
#define _IO_WXSGETN(FP, DATA, N) WJUMP2 (__xsgetn, FP, DATA, N)

/* The 'seekoff' hook moves the stream position to a new position
   relative to the start of the file (if DIR==0), the current position
   (MODE==1), or the end of the file (MODE==2).
   It matches the streambuf::seekoff virtual function.
   It is also used for the ANSI fseek function. */
typedef off64_t (*_IO_seekoff_t) (FILE *FP, off64_t OFF, int DIR,
				      int MODE);
#define _IO_SEEKOFF(FP, OFF, DIR, MODE) JUMP3 (__seekoff, FP, OFF, DIR, MODE)
#define _IO_WSEEKOFF(FP, OFF, DIR, MODE) WJUMP3 (__seekoff, FP, OFF, DIR, MODE)

/* The 'seekpos' hook also moves the stream position,
   but to an absolute position given by a fpos64_t (seekpos).
   It matches the streambuf::seekpos virtual function.
   It is also used for the ANSI fgetpos and fsetpos functions.  */
/* The _IO_seek_cur and _IO_seek_end options are not allowed. */
typedef off64_t (*_IO_seekpos_t) (FILE *, off64_t, int);
#define _IO_SEEKPOS(FP, POS, FLAGS) JUMP2 (__seekpos, FP, POS, FLAGS)
#define _IO_WSEEKPOS(FP, POS, FLAGS) WJUMP2 (__seekpos, FP, POS, FLAGS)

/* The 'setbuf' hook gives a buffer to the file.
   It matches the streambuf::setbuf virtual function. */
typedef FILE* (*_IO_setbuf_t) (FILE *, char *, ssize_t);
#define _IO_SETBUF(FP, BUFFER, LENGTH) JUMP2 (__setbuf, FP, BUFFER, LENGTH)
#define _IO_WSETBUF(FP, BUFFER, LENGTH) WJUMP2 (__setbuf, FP, BUFFER, LENGTH)

/* The 'sync' hook attempts to synchronize the internal data structures
   of the file with the external state.
   It matches the streambuf::sync virtual function. */
typedef int (*_IO_sync_t) (FILE *);
#define _IO_SYNC(FP) JUMP0 (__sync, FP)
#define _IO_WSYNC(FP) WJUMP0 (__sync, FP)

/* The 'doallocate' hook is used to tell the file to allocate a buffer.
   It matches the streambuf::doallocate virtual function, which is not
   in the ANSI/ISO C++ standard, but is part traditional implementations. */
typedef int (*_IO_doallocate_t) (FILE *);
#define _IO_DOALLOCATE(FP) JUMP0 (__doallocate, FP)
#define _IO_WDOALLOCATE(FP) WJUMP0 (__doallocate, FP)

/* The following four hooks (sysread, syswrite, sysclose, sysseek, and
   sysstat) are low-level hooks specific to this implementation.
   There is no correspondence in the ANSI/ISO C++ standard library.
   The hooks basically correspond to the Unix system functions
   (read, write, close, lseek, and stat) except that a FILE*
   parameter is used instead of an integer file descriptor;  the default
   implementation used for normal files just calls those functions.
   The advantage of overriding these functions instead of the higher-level
   ones (underflow, overflow etc) is that you can leave all the buffering
   higher-level functions.  */

/* The 'sysread' hook is used to read data from the external file into
   an existing buffer.  It generalizes the Unix read(2) function.
   It matches the streambuf::sys_read virtual function, which is
   specific to this implementation. */
typedef ssize_t (*_IO_read_t) (FILE *, void *, ssize_t);
#define _IO_SYSREAD(FP, DATA, LEN) JUMP2 (__read, FP, DATA, LEN)
#define _IO_WSYSREAD(FP, DATA, LEN) WJUMP2 (__read, FP, DATA, LEN)

/* The 'syswrite' hook is used to write data from an existing buffer
   to an external file.  It generalizes the Unix write(2) function.
   It matches the streambuf::sys_write virtual function, which is
   specific to this implementation. */
typedef ssize_t (*_IO_write_t) (FILE *, const void *, ssize_t);
#define _IO_SYSWRITE(FP, DATA, LEN) JUMP2 (__write, FP, DATA, LEN)
#define _IO_WSYSWRITE(FP, DATA, LEN) WJUMP2 (__write, FP, DATA, LEN)

/* The 'sysseek' hook is used to re-position an external file.
   It generalizes the Unix lseek(2) function.
   It matches the streambuf::sys_seek virtual function, which is
   specific to this implementation. */
typedef off64_t (*_IO_seek_t) (FILE *, off64_t, int);
#define _IO_SYSSEEK(FP, OFFSET, MODE) JUMP2 (__seek, FP, OFFSET, MODE)
#define _IO_WSYSSEEK(FP, OFFSET, MODE) WJUMP2 (__seek, FP, OFFSET, MODE)

/* The 'sysclose' hook is used to finalize (close, finish up) an
   external file.  It generalizes the Unix close(2) function.
   It matches the streambuf::sys_close virtual function, which is
   specific to this implementation. */
typedef int (*_IO_close_t) (FILE *); /* finalize */
#define _IO_SYSCLOSE(FP) JUMP0 (__close, FP)
#define _IO_WSYSCLOSE(FP) WJUMP0 (__close, FP)

/* The 'sysstat' hook is used to get information about an external file
   into a struct stat buffer.  It generalizes the Unix fstat(2) call.
   It matches the streambuf::sys_stat virtual function, which is
   specific to this implementation. */
typedef int (*_IO_stat_t) (FILE *, void *);
#define _IO_SYSSTAT(FP, BUF) JUMP1 (__stat, FP, BUF)
#define _IO_WSYSSTAT(FP, BUF) WJUMP1 (__stat, FP, BUF)

/* The 'showmany' hook can be used to get an image how much input is
   available.  In many cases the answer will be 0 which means unknown
   but some cases one can provide real information.  */
typedef int (*_IO_showmanyc_t) (FILE *);
#define _IO_SHOWMANYC(FP) JUMP0 (__showmanyc, FP)
#define _IO_WSHOWMANYC(FP) WJUMP0 (__showmanyc, FP)

/* The 'imbue' hook is used to get information about the currently
   installed locales.  */
typedef void (*_IO_imbue_t) (FILE *, void *);
#define _IO_IMBUE(FP, LOCALE) JUMP1 (__imbue, FP, LOCALE)
#define _IO_WIMBUE(FP, LOCALE) WJUMP1 (__imbue, FP, LOCALE)


#define _IO_CHAR_TYPE char /* unsigned char ? */
#define _IO_INT_TYPE int

struct _IO_jump_t
{
    JUMP_FIELD(size_t, __dummy);
    JUMP_FIELD(size_t, __dummy2);
    JUMP_FIELD(_IO_finish_t, __finish);
    JUMP_FIELD(_IO_overflow_t, __overflow);
    JUMP_FIELD(_IO_underflow_t, __underflow);
    JUMP_FIELD(_IO_underflow_t, __uflow);
    JUMP_FIELD(_IO_pbackfail_t, __pbackfail);
    /* showmany */
    JUMP_FIELD(_IO_xsputn_t, __xsputn);
    JUMP_FIELD(_IO_xsgetn_t, __xsgetn);
    JUMP_FIELD(_IO_seekoff_t, __seekoff);
    JUMP_FIELD(_IO_seekpos_t, __seekpos);
    JUMP_FIELD(_IO_setbuf_t, __setbuf);
    JUMP_FIELD(_IO_sync_t, __sync);
    JUMP_FIELD(_IO_doallocate_t, __doallocate);
    JUMP_FIELD(_IO_read_t, __read);
    JUMP_FIELD(_IO_write_t, __write);
    JUMP_FIELD(_IO_seek_t, __seek);
    JUMP_FIELD(_IO_close_t, __close);
    JUMP_FIELD(_IO_stat_t, __stat);
    JUMP_FIELD(_IO_showmanyc_t, __showmanyc);
    JUMP_FIELD(_IO_imbue_t, __imbue);
};

/* We always allocate an extra word following an _IO_FILE.
   This contains a pointer to the function jump table used.
   This is for compatibility with C++ streambuf; the word can
   be used to smash to a pointer to a virtual function table. */

struct _IO_FILE_plus
{
  FILE file;
  const struct _IO_jump_t *vtable;
};

#ifdef _IO_USE_OLD_IO_FILE
/* This structure is used by the compatibility code as if it were an
   _IO_FILE_plus, but has enough space to initialize the _mode argument
   of an _IO_FILE_complete.  */
struct _IO_FILE_complete_plus
{
  struct _IO_FILE_complete file;
  const struct _IO_jump_t *vtable;
};
#endif

/* Special file type for fopencookie function.  */
struct _IO_cookie_file
{
  struct _IO_FILE_plus __fp;
  void *__cookie;
  cookie_io_functions_t __io_functions;
};

FILE *_IO_fopencookie (void *cookie, const char *mode,
                       cookie_io_functions_t io_functions);


/* Iterator type for walking global linked list of _IO_FILE objects. */

typedef FILE *_IO_ITER;

/* Generic functions */

extern void _IO_switch_to_main_get_area (FILE *) __THROW;
extern void _IO_switch_to_backup_area (FILE *) __THROW;
extern int _IO_switch_to_get_mode (FILE *);
libc_hidden_proto (_IO_switch_to_get_mode)
extern void _IO_init_internal (FILE *, int) attribute_hidden;
extern int _IO_sputbackc (FILE *, int) __THROW;
libc_hidden_proto (_IO_sputbackc)
extern int _IO_sungetc (FILE *) __THROW;
extern void _IO_un_link (struct _IO_FILE_plus *) __THROW;
libc_hidden_proto (_IO_un_link)
extern void _IO_link_in (struct _IO_FILE_plus *) __THROW;
libc_hidden_proto (_IO_link_in)
extern void _IO_doallocbuf (FILE *) __THROW;
libc_hidden_proto (_IO_doallocbuf)
extern void _IO_unsave_markers (FILE *) __THROW;
libc_hidden_proto (_IO_unsave_markers)
extern void _IO_setb (FILE *, char *, char *, int) __THROW;
libc_hidden_proto (_IO_setb)
extern unsigned _IO_adjust_column (unsigned, const char *, int) __THROW;
libc_hidden_proto (_IO_adjust_column)
#define _IO_sputn(__fp, __s, __n) _IO_XSPUTN (__fp, __s, __n)

ssize_t _IO_least_wmarker (FILE *, wchar_t *) __THROW;
libc_hidden_proto (_IO_least_wmarker)
extern void _IO_switch_to_main_wget_area (FILE *) __THROW;
libc_hidden_proto (_IO_switch_to_main_wget_area)
extern void _IO_switch_to_wbackup_area (FILE *) __THROW;
libc_hidden_proto (_IO_switch_to_wbackup_area)
extern int _IO_switch_to_wget_mode (FILE *);
libc_hidden_proto (_IO_switch_to_wget_mode)
extern void _IO_wsetb (FILE *, wchar_t *, wchar_t *, int) __THROW;
libc_hidden_proto (_IO_wsetb)
extern wint_t _IO_sputbackwc (FILE *, wint_t) __THROW;
libc_hidden_proto (_IO_sputbackwc)
extern wint_t _IO_sungetwc (FILE *) __THROW;
extern void _IO_wdoallocbuf (FILE *) __THROW;
libc_hidden_proto (_IO_wdoallocbuf)
extern void _IO_unsave_wmarkers (FILE *) __THROW;
extern unsigned _IO_adjust_wcolumn (unsigned, const wchar_t *, int) __THROW;
extern off64_t get_file_offset (FILE *fp);

/* Marker-related function. */

extern void _IO_init_marker (struct _IO_marker *, FILE *);
extern void _IO_init_wmarker (struct _IO_marker *, FILE *);
extern void _IO_remove_marker (struct _IO_marker *) __THROW;
extern int _IO_marker_difference (struct _IO_marker *, struct _IO_marker *)
     __THROW;
extern int _IO_marker_delta (struct _IO_marker *) __THROW;
extern int _IO_wmarker_delta (struct _IO_marker *) __THROW;
extern int _IO_seekmark (FILE *, struct _IO_marker *, int) __THROW;
extern int _IO_seekwmark (FILE *, struct _IO_marker *, int) __THROW;

/* Functions for iterating global list and dealing with its lock */

extern _IO_ITER _IO_iter_begin (void) __THROW;
libc_hidden_proto (_IO_iter_begin)
extern _IO_ITER _IO_iter_end (void) __THROW;
libc_hidden_proto (_IO_iter_end)
extern _IO_ITER _IO_iter_next (_IO_ITER) __THROW;
libc_hidden_proto (_IO_iter_next)
extern FILE *_IO_iter_file (_IO_ITER) __THROW;
libc_hidden_proto (_IO_iter_file)
extern void _IO_list_lock (void) __THROW;
libc_hidden_proto (_IO_list_lock)
extern void _IO_list_unlock (void) __THROW;
libc_hidden_proto (_IO_list_unlock)
extern void _IO_list_resetlock (void) __THROW;
libc_hidden_proto (_IO_list_resetlock)
extern void _IO_enable_locks (void) __THROW;
libc_hidden_proto (_IO_enable_locks)

/* Default jumptable functions. */

extern int _IO_default_underflow (FILE *) __THROW;
extern int _IO_default_uflow (FILE *);
libc_hidden_proto (_IO_default_uflow)
extern wint_t _IO_wdefault_uflow (FILE *);
libc_hidden_proto (_IO_wdefault_uflow)
extern int _IO_default_doallocate (FILE *) __THROW;
libc_hidden_proto (_IO_default_doallocate)
extern int _IO_wdefault_doallocate (FILE *) __THROW;
libc_hidden_proto (_IO_wdefault_doallocate)
extern void _IO_default_finish (FILE *, int) __THROW;
libc_hidden_proto (_IO_default_finish)
extern void _IO_wdefault_finish (FILE *, int) __THROW;
libc_hidden_proto (_IO_wdefault_finish)
extern int _IO_default_pbackfail (FILE *, int) __THROW;
libc_hidden_proto (_IO_default_pbackfail)
extern wint_t _IO_wdefault_pbackfail (FILE *, wint_t) __THROW;
libc_hidden_proto (_IO_wdefault_pbackfail)
extern FILE* _IO_default_setbuf (FILE *, char *, ssize_t);
extern size_t _IO_default_xsputn (FILE *, const void *, size_t);
libc_hidden_proto (_IO_default_xsputn)
extern size_t _IO_wdefault_xsputn (FILE *, const void *, size_t);
libc_hidden_proto (_IO_wdefault_xsputn)
extern size_t _IO_default_xsgetn (FILE *, void *, size_t);
libc_hidden_proto (_IO_default_xsgetn)
extern size_t _IO_wdefault_xsgetn (FILE *, void *, size_t);
libc_hidden_proto (_IO_wdefault_xsgetn)
extern off64_t _IO_default_seekoff (FILE *, off64_t, int, int)
     __THROW;
extern off64_t _IO_default_seekpos (FILE *, off64_t, int);
extern ssize_t _IO_default_write (FILE *, const void *, ssize_t);
extern ssize_t _IO_default_read (FILE *, void *, ssize_t);
extern int _IO_default_stat (FILE *, void *) __THROW;
extern off64_t _IO_default_seek (FILE *, off64_t, int) __THROW;
extern int _IO_default_sync (FILE *) __THROW;
#define _IO_default_close ((_IO_close_t) _IO_default_sync)
extern int _IO_default_showmanyc (FILE *) __THROW;
extern void _IO_default_imbue (FILE *, void *) __THROW;

extern const struct _IO_jump_t _IO_file_jumps;
libc_hidden_proto (_IO_file_jumps)
extern const struct _IO_jump_t _IO_file_jumps_mmap attribute_hidden;
extern const struct _IO_jump_t _IO_file_jumps_maybe_mmap attribute_hidden;
extern const struct _IO_jump_t _IO_wfile_jumps;
libc_hidden_proto (_IO_wfile_jumps)
extern const struct _IO_jump_t _IO_wfile_jumps_mmap attribute_hidden;
extern const struct _IO_jump_t _IO_wfile_jumps_maybe_mmap attribute_hidden;
extern const struct _IO_jump_t _IO_old_file_jumps attribute_hidden;
extern const struct _IO_jump_t _IO_streambuf_jumps;
extern const struct _IO_jump_t _IO_old_proc_jumps attribute_hidden;
extern const struct _IO_jump_t _IO_str_jumps attribute_hidden;
extern const struct _IO_jump_t _IO_wstr_jumps attribute_hidden;
extern int _IO_do_write (FILE *, const char *, size_t);
libc_hidden_proto (_IO_do_write)
extern int _IO_new_do_write (FILE *, const char *, size_t);
extern int _IO_old_do_write (FILE *, const char *, size_t);
extern int _IO_wdo_write (FILE *, const wchar_t *, size_t);
libc_hidden_proto (_IO_wdo_write)
extern int _IO_flush_all_lockp (int);
extern int _IO_flush_all (void);
libc_hidden_proto (_IO_flush_all)
extern int _IO_cleanup (void);
extern void _IO_flush_all_linebuffered (void);
libc_hidden_proto (_IO_flush_all_linebuffered)
extern int _IO_new_fgetpos (FILE *, __fpos_t *);
extern int _IO_old_fgetpos (FILE *, __fpos_t *);
extern int _IO_new_fsetpos (FILE *, const __fpos_t *);
extern int _IO_old_fsetpos (FILE *, const __fpos_t *);
extern int _IO_new_fgetpos64 (FILE *, __fpos64_t *);
extern int _IO_old_fgetpos64 (FILE *, __fpos64_t *);
extern int _IO_new_fsetpos64 (FILE *, const __fpos64_t *);
extern int _IO_old_fsetpos64 (FILE *, const __fpos64_t *);
extern void _IO_old_init (FILE *fp, int flags) __THROW;


#define _IO_do_flush(_f) \
  ((_f)->_mode <= 0							      \
   ? _IO_do_write(_f, (_f)->_IO_write_base,				      \
		  (_f)->_IO_write_ptr-(_f)->_IO_write_base)		      \
   : _IO_wdo_write(_f, (_f)->_wide_data->_IO_write_base,		      \
		   ((_f)->_wide_data->_IO_write_ptr			      \
		    - (_f)->_wide_data->_IO_write_base)))
#define _IO_old_do_flush(_f) \
  _IO_old_do_write(_f, (_f)->_IO_write_base, \
		   (_f)->_IO_write_ptr-(_f)->_IO_write_base)
#define _IO_in_put_mode(_fp) ((_fp)->_flags & _IO_CURRENTLY_PUTTING)
#define _IO_mask_flags(fp, f, mask) \
       ((fp)->_flags = ((fp)->_flags & ~(mask)) | ((f) & (mask)))
#define _IO_setg(fp, eb, g, eg)  ((fp)->_IO_read_base = (eb),\
	(fp)->_IO_read_ptr = (g), (fp)->_IO_read_end = (eg))
#define _IO_wsetg(fp, eb, g, eg)  ((fp)->_wide_data->_IO_read_base = (eb),\
	(fp)->_wide_data->_IO_read_ptr = (g), \
	(fp)->_wide_data->_IO_read_end = (eg))
#define _IO_setp(__fp, __p, __ep) \
       ((__fp)->_IO_write_base = (__fp)->_IO_write_ptr \
	= __p, (__fp)->_IO_write_end = (__ep))
#define _IO_wsetp(__fp, __p, __ep) \
       ((__fp)->_wide_data->_IO_write_base \
	= (__fp)->_wide_data->_IO_write_ptr = __p, \
	(__fp)->_wide_data->_IO_write_end = (__ep))
#define _IO_have_backup(fp) ((fp)->_IO_save_base != NULL)
#define _IO_have_wbackup(fp) ((fp)->_wide_data->_IO_save_base != NULL)
#define _IO_in_backup(fp) ((fp)->_flags & _IO_IN_BACKUP)
#define _IO_have_markers(fp) ((fp)->_markers != NULL)
#define _IO_blen(fp) ((fp)->_IO_buf_end - (fp)->_IO_buf_base)
#define _IO_wblen(fp) ((fp)->_wide_data->_IO_buf_end \
		       - (fp)->_wide_data->_IO_buf_base)

/* Jumptable functions for files. */

extern int _IO_file_doallocate (FILE *) __THROW;
libc_hidden_proto (_IO_file_doallocate)
extern FILE* _IO_file_setbuf (FILE *, char *, ssize_t);
libc_hidden_proto (_IO_file_setbuf)
extern off64_t _IO_file_seekoff (FILE *, off64_t, int, int);
libc_hidden_proto (_IO_file_seekoff)
extern off64_t _IO_file_seekoff_mmap (FILE *, off64_t, int, int)
     __THROW;
extern size_t _IO_file_xsputn (FILE *, const void *, size_t);
libc_hidden_proto (_IO_file_xsputn)
extern size_t _IO_file_xsgetn (FILE *, void *, size_t);
libc_hidden_proto (_IO_file_xsgetn)
extern int _IO_file_stat (FILE *, void *) __THROW;
libc_hidden_proto (_IO_file_stat)
extern int _IO_file_close (FILE *) __THROW;
libc_hidden_proto (_IO_file_close)
extern int _IO_file_close_mmap (FILE *) __THROW;
extern int _IO_file_underflow (FILE *);
libc_hidden_proto (_IO_file_underflow)
extern int _IO_file_underflow_mmap (FILE *);
extern int _IO_file_underflow_maybe_mmap (FILE *);
extern int _IO_file_overflow (FILE *, int);
libc_hidden_proto (_IO_file_overflow)
#define _IO_file_is_open(__fp) ((__fp)->_fileno != -1)
extern FILE* _IO_file_attach (FILE *, int);
libc_hidden_proto (_IO_file_attach)
extern FILE* _IO_file_open (FILE *, const char *, int, int, int, int);
libc_hidden_proto (_IO_file_open)
extern FILE* _IO_file_fopen (FILE *, const char *, const char *, int);
libc_hidden_proto (_IO_file_fopen)
extern ssize_t _IO_file_write (FILE *, const void *, ssize_t);
extern ssize_t _IO_file_read (FILE *, void *, ssize_t);
libc_hidden_proto (_IO_file_read)
extern int _IO_file_sync (FILE *);
libc_hidden_proto (_IO_file_sync)
extern int _IO_file_close_it (FILE *);
libc_hidden_proto (_IO_file_close_it)
extern off64_t _IO_file_seek (FILE *, off64_t, int) __THROW;
libc_hidden_proto (_IO_file_seek)
extern void _IO_file_finish (FILE *, int);
libc_hidden_proto (_IO_file_finish)

extern FILE* _IO_new_file_attach (FILE *, int);
extern int _IO_new_file_close_it (FILE *);
extern void _IO_new_file_finish (FILE *, int);
extern FILE* _IO_new_file_fopen (FILE *, const char *, const char *,
				     int);
extern void _IO_no_init (FILE *, int, int, struct _IO_wide_data *,
			 const struct _IO_jump_t *) __THROW;
extern void _IO_new_file_init_internal (struct _IO_FILE_plus *)
  __THROW attribute_hidden;
extern FILE* _IO_new_file_setbuf (FILE *, char *, ssize_t);
extern FILE* _IO_file_setbuf_mmap (FILE *, char *, ssize_t);
extern int _IO_new_file_sync (FILE *);
extern int _IO_new_file_underflow (FILE *);
extern int _IO_new_file_overflow (FILE *, int);
extern off64_t _IO_new_file_seekoff (FILE *, off64_t, int, int);
extern ssize_t _IO_new_file_write (FILE *, const void *, ssize_t);
extern size_t _IO_new_file_xsputn (FILE *, const void *, size_t);

extern FILE* _IO_old_file_setbuf (FILE *, char *, ssize_t);
extern off64_t _IO_old_file_seekoff (FILE *, off64_t, int, int);
extern size_t _IO_old_file_xsputn (FILE *, const void *, size_t);
extern int _IO_old_file_underflow (FILE *);
extern int _IO_old_file_overflow (FILE *, int);
extern void _IO_old_file_init_internal (struct _IO_FILE_plus *)
  __THROW attribute_hidden;
extern FILE* _IO_old_file_attach (FILE *, int);
extern FILE* _IO_old_file_fopen (FILE *, const char *, const char *);
extern ssize_t _IO_old_file_write (FILE *, const void *, ssize_t);
extern int _IO_old_file_sync (FILE *);
extern int _IO_old_file_close_it (FILE *);
extern void _IO_old_file_finish (FILE *, int);

extern int _IO_wfile_doallocate (FILE *) __THROW;
extern size_t _IO_wfile_xsputn (FILE *, const void *, size_t);
libc_hidden_proto (_IO_wfile_xsputn)
extern FILE* _IO_wfile_setbuf (FILE *, wchar_t *, ssize_t);
extern wint_t _IO_wfile_sync (FILE *);
libc_hidden_proto (_IO_wfile_sync)
extern wint_t _IO_wfile_underflow (FILE *);
libc_hidden_proto (_IO_wfile_underflow)
extern wint_t _IO_wfile_overflow (FILE *, wint_t);
libc_hidden_proto (_IO_wfile_overflow)
extern off64_t _IO_wfile_seekoff (FILE *, off64_t, int, int);
libc_hidden_proto (_IO_wfile_seekoff)

/* Jumptable functions for proc_files. */
extern FILE* _IO_proc_open (FILE *, const char *, const char *)
     __THROW;
extern FILE* _IO_new_proc_open (FILE *, const char *, const char *)
     __THROW;
extern FILE* _IO_old_proc_open (FILE *, const char *, const char *);
extern int _IO_proc_close (FILE *) __THROW;
extern int _IO_new_proc_close (FILE *) __THROW;
extern int _IO_old_proc_close (FILE *);

/* Jumptable functions for strfiles. */
extern int _IO_str_underflow (FILE *) __THROW;
libc_hidden_proto (_IO_str_underflow)
extern int _IO_str_overflow (FILE *, int) __THROW;
libc_hidden_proto (_IO_str_overflow)
extern int _IO_str_pbackfail (FILE *, int) __THROW;
libc_hidden_proto (_IO_str_pbackfail)
extern off64_t _IO_str_seekoff (FILE *, off64_t, int, int) __THROW;
libc_hidden_proto (_IO_str_seekoff)
extern void _IO_str_finish (FILE *, int) __THROW;

/* Other strfile functions */
struct _IO_strfile_;
extern ssize_t _IO_str_count (FILE *) __THROW;

/* And the wide character versions.  */
extern void _IO_wstr_init_static (FILE *, wchar_t *, size_t, wchar_t *)
     __THROW;
extern ssize_t _IO_wstr_count (FILE *) __THROW;
extern wint_t _IO_wstr_overflow (FILE *, wint_t) __THROW;
extern wint_t _IO_wstr_underflow (FILE *) __THROW;
extern off64_t _IO_wstr_seekoff (FILE *, off64_t, int, int)
     __THROW;
extern wint_t _IO_wstr_pbackfail (FILE *, wint_t) __THROW;
extern void _IO_wstr_finish (FILE *, int) __THROW;

/* Internal versions of v*printf that take an additional flags
   parameter.  */
extern int __vfprintf_internal (FILE *fp, const char *format, va_list ap,
				unsigned int mode_flags)
    attribute_hidden;
extern int __vfwprintf_internal (FILE *fp, const wchar_t *format, va_list ap,
				 unsigned int mode_flags)
    attribute_hidden;

extern int __vasprintf_internal (char **result_ptr, const char *format,
				 va_list ap, unsigned int mode_flags)
    attribute_hidden;
extern int __vdprintf_internal (int d, const char *format, va_list ap,
				unsigned int mode_flags)
    attribute_hidden;
extern int __obstack_vprintf_internal (struct obstack *ob, const char *fmt,
				       va_list ap, unsigned int mode_flags)
    attribute_hidden;

/* Note: __vsprintf_internal, unlike vsprintf, does take a maxlen argument,
   because it's called by both vsprintf and vsprintf_chk.  If maxlen is
   not set to -1, overrunning the buffer will cause a prompt crash.
   This is the behavior of ordinary (v)sprintf functions, thus they call
   __vsprintf_internal with that argument set to -1.  */
extern int __vsprintf_internal (char *string, size_t maxlen,
				const char *format, va_list ap,
				unsigned int mode_flags)
    attribute_hidden;

extern int __vsnprintf_internal (char *string, size_t maxlen,
				 const char *format, va_list ap,
				 unsigned int mode_flags)
    attribute_hidden;
extern int __vswprintf_internal (wchar_t *string, size_t maxlen,
				 const wchar_t *format, va_list ap,
				 unsigned int mode_flags)
    attribute_hidden;

/* Flags for __v*printf_internal.

   PRINTF_LDBL_IS_DBL indicates whether long double values are to be
   handled as having the same format as double, in which case the flag
   should be set to one, or as another format, otherwise.

   PRINTF_FORTIFY, when set to one, indicates that fortification checks
   are to be performed in input parameters.  This is used by the
   __*printf_chk functions, which are used when _FORTIFY_SOURCE is
   defined to 1 or 2.  Otherwise, such checks are ignored.

   PRINTF_CHK indicates, to the internal function being called, that the
   call is originated from one of the __*printf_chk functions.

   PRINTF_LDBL_USES_FLOAT128 is used on platforms where the long double
   format used to be different from the IEC 60559 double format *and*
   also different from the Quadruple 128-bits IEC 60559 format (such as
   the IBM Extended Precision format on powerpc or the 80-bits IEC 60559
   format on x86), but was later converted to the Quadruple 128-bits IEC
   60559 format, which is the same format that the _Float128 always has
   (hence the `USES_FLOAT128' suffix in the name of the flag).  When set
   to one, this macro indicates that long double values are to be
   handled as having this new format.  Otherwise, they should be handled
   as the previous format on that platform.  */
#define PRINTF_LDBL_IS_DBL		0x0001
#define PRINTF_FORTIFY			0x0002
#define PRINTF_CHK			0x0004
#define PRINTF_LDBL_USES_FLOAT128	0x0008

extern size_t _IO_getline (FILE *,char *, size_t, int, int);
libc_hidden_proto (_IO_getline)
extern size_t _IO_getline_info (FILE *,char *, size_t,
				    int, int, int *);
libc_hidden_proto (_IO_getline_info)
extern size_t _IO_getwline (FILE *,wchar_t *, size_t, wint_t, int);
extern size_t _IO_getwline_info (FILE *,wchar_t *, size_t,
				     wint_t, int, wint_t *);

extern struct _IO_FILE_plus *_IO_list_all;
libc_hidden_proto (_IO_list_all)
extern void (*_IO_cleanup_registration_needed) (void);

extern void _IO_str_init_static_internal (struct _IO_strfile_ *, char *,
					  size_t, char *) __THROW;
extern off64_t _IO_seekoff_unlocked (FILE *, off64_t, int, int)
     attribute_hidden;
extern off64_t _IO_seekpos_unlocked (FILE *, off64_t, int)
     attribute_hidden;

#if _G_HAVE_MMAP

# include <unistd.h>
# include <fcntl.h>
# include <sys/mman.h>
# include <sys/param.h>

# if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#  define MAP_ANONYMOUS MAP_ANON
# endif

# if !defined(MAP_ANONYMOUS) || !defined(EXEC_PAGESIZE)
#  undef _G_HAVE_MMAP
#  define _G_HAVE_MMAP 0
# endif

#endif /* _G_HAVE_MMAP */

/* Flags for __vfscanf_internal and __vfwscanf_internal.

   SCANF_LDBL_IS_DBL indicates whether long double values are to be
   handled as having the same format as double, in which case the flag
   should be set to one, or as another format, otherwise.

   SCANF_ISOC99_A, when set to one, indicates that the ISO C99 or POSIX
   behavior of the scanf functions is to be used, i.e. automatic
   allocation for input strings with %as, %aS and %a[, a GNU extension,
   is disabled. This is the behavior that the __isoc99_scanf family of
   functions use.  When the flag is set to zero, automatic allocation is
   enabled.

   SCANF_LDBL_USES_FLOAT128 is used on platforms where the long double
   format used to be different from the IEC 60559 double format *and*
   also different from the Quadruple 128-bits IEC 60559 format (such as
   the IBM Extended Precision format on powerpc or the 80-bits IEC 60559
   format on x86), but was later converted to the Quadruple 128-bits IEC
   60559 format, which is the same format that the _Float128 always has
   (hence the `USES_FLOAT128' suffix in the name of the flag).  When set
   to one, this macros indicates that long double values are to be
   handled as having this new format.  Otherwise, they should be handled
   as the previous format on that platform.  */
#define SCANF_LDBL_IS_DBL		0x0001
#define SCANF_ISOC99_A			0x0002
#define SCANF_LDBL_USES_FLOAT128	0x0004

extern int __vfscanf_internal (FILE *fp, const char *format, va_list argp,
			       unsigned int flags)
  attribute_hidden;
extern int __vfwscanf_internal (FILE *fp, const wchar_t *format, va_list argp,
				unsigned int flags)
  attribute_hidden;

extern int _IO_vscanf (const char *, va_list) __THROW;

#ifdef _IO_MTSAFE_IO
/* check following! */
# ifdef _IO_USE_OLD_IO_FILE
#  define FILEBUF_LITERAL(CHAIN, FLAGS, FD, WDP) \
       { _IO_MAGIC+_IO_LINKED+_IO_IS_FILEBUF+FLAGS, \
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (FILE *) CHAIN, FD, \
	 0, _IO_pos_BAD, 0, 0, { 0 }, &_IO_stdfile_##FD##_lock }
# else
#  define FILEBUF_LITERAL(CHAIN, FLAGS, FD, WDP) \
       { _IO_MAGIC+_IO_LINKED+_IO_IS_FILEBUF+FLAGS, \
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (FILE *) CHAIN, FD, \
	 0, _IO_pos_BAD, 0, 0, { 0 }, &_IO_stdfile_##FD##_lock, _IO_pos_BAD,\
	 NULL, WDP, 0 }
# endif
#else
# ifdef _IO_USE_OLD_IO_FILE
#  define FILEBUF_LITERAL(CHAIN, FLAGS, FD, WDP) \
       { _IO_MAGIC+_IO_LINKED+_IO_IS_FILEBUF+FLAGS, \
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (FILE *) CHAIN, FD, \
	 0, _IO_pos_BAD }
# else
#  define FILEBUF_LITERAL(CHAIN, FLAGS, FD, WDP) \
       { _IO_MAGIC+_IO_LINKED+_IO_IS_FILEBUF+FLAGS, \
	 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (FILE *) CHAIN, FD, \
	 0, _IO_pos_BAD, 0, 0, { 0 }, 0, _IO_pos_BAD, \
	 NULL, WDP, 0 }
# endif
#endif

#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
/* See oldstdfiles.c.  These are the old stream variables.  */
extern struct _IO_FILE_plus _IO_stdin_;
extern struct _IO_FILE_plus _IO_stdout_;
extern struct _IO_FILE_plus _IO_stderr_;

static inline bool
_IO_legacy_file (FILE *fp)
{
  return fp == (FILE *) &_IO_stdin_ || fp == (FILE *) &_IO_stdout_
    || fp == (FILE *) &_IO_stderr_;
}
#endif

/* Deallocate a stream if it is heap-allocated.  Preallocated
   stdin/stdout/stderr streams are not deallocated. */
static inline void
_IO_deallocate_file (FILE *fp)
{
  /* The current stream variables.  */
  if (fp == (FILE *) &_IO_2_1_stdin_ || fp == (FILE *) &_IO_2_1_stdout_
      || fp == (FILE *) &_IO_2_1_stderr_)
    return;
#if SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_1)
  if (_IO_legacy_file (fp))
    return;
#endif
  free (fp);
}

#ifdef IO_DEBUG
# define CHECK_FILE(FILE, RET) do {				\
    if ((FILE) == NULL						\
	|| ((FILE)->_flags & _IO_MAGIC_MASK) != _IO_MAGIC)	\
      {								\
	__set_errno (EINVAL);					\
	return RET;						\
      }								\
  } while (0)
#else
# define CHECK_FILE(FILE, RET) do { } while (0)
#endif

static inline void
__attribute__ ((__always_inline__))
_IO_acquire_lock_fct (FILE **p)
{
  FILE *fp = *p;
  if ((fp->_flags & _IO_USER_LOCK) == 0)
    _IO_funlockfile (fp);
}

#if !defined _IO_MTSAFE_IO && IS_IN (libc)
# define _IO_acquire_lock(_fp)						      \
  do {
# define _IO_release_lock(_fp)						      \
  } while (0)
#endif

/* Collect all vtables in a special section for vtable verification.
   These symbols cover the extent of this section.  */
symbol_set_declare (__libc_IO_vtables)

/* libio vtables need to carry this attribute so that they pass
   validation.  */
#define libio_vtable __attribute__ ((section ("__libc_IO_vtables")))

#ifdef SHARED
/* If equal to &_IO_vtable_check (with pointer guard protection),
   unknown vtable pointers are valid.  This function pointer is solely
   used as a flag.  */
extern void (*IO_accept_foreign_vtables) (void) attribute_hidden;

/* Assigns the passed function pointer (either NULL or
   &_IO_vtable_check) to IO_accept_foreign_vtables.  */
static inline void
IO_set_accept_foreign_vtables (void (*flag) (void))
{
#ifdef PTR_MANGLE
  PTR_MANGLE (flag);
#endif
  atomic_store_relaxed (&IO_accept_foreign_vtables, flag);
}

#else  /* !SHARED */

/* The statically-linked version does nothing. */
static inline void
IO_set_accept_foreign_vtables (void (*flag) (void))
{
}

#endif

/* Check if unknown vtable pointers are permitted; otherwise,
   terminate the process.  */
void _IO_vtable_check (void) attribute_hidden;

/* Perform vtable pointer validation.  If validation fails, terminate
   the process.  */
static inline const struct _IO_jump_t *
IO_validate_vtable (const struct _IO_jump_t *vtable)
{
  /* Fast path: The vtable pointer is within the __libc_IO_vtables
     section.  */
  uintptr_t section_length = __stop___libc_IO_vtables - __start___libc_IO_vtables;
  uintptr_t ptr = (uintptr_t) vtable;
  uintptr_t offset = ptr - (uintptr_t) __start___libc_IO_vtables;
  if (__glibc_unlikely (offset >= section_length))
    /* The vtable pointer is not in the expected section.  Use the
       slow path, which will terminate the process if necessary.  */
    _IO_vtable_check ();
  return vtable;
}

/* Character set conversion.  */

enum __codecvt_result
{
  __codecvt_ok,
  __codecvt_partial,
  __codecvt_error,
  __codecvt_noconv
};

enum __codecvt_result __libio_codecvt_out (struct _IO_codecvt *,
					   __mbstate_t *,
					   const wchar_t *,
					   const wchar_t *,
					   const wchar_t **, char *,
					   char *, char **)
  attribute_hidden;
enum __codecvt_result __libio_codecvt_in (struct _IO_codecvt *,
					  __mbstate_t *,
					  const char *, const char *,
					  const char **, wchar_t *,
					  wchar_t *, wchar_t **)
  attribute_hidden;
int __libio_codecvt_encoding (struct _IO_codecvt *) attribute_hidden;
int __libio_codecvt_length (struct _IO_codecvt *, __mbstate_t *,
			    const char *, const char *, size_t)
  attribute_hidden;

#endif /* libioP.h.  */
