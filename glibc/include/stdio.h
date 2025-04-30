#ifndef _STDIO_H
# if !defined _ISOMAC && defined _IO_MTSAFE_IO
#  include <stdio-lock.h>
# endif

/* Workaround PR90731 with GCC 9 when using ldbl redirects in C++.  */
# include <bits/floatn.h>
# if defined __cplusplus && __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
#  if __GNUC_PREREQ (9, 0) && !__GNUC_PREREQ (9, 3)
#    pragma GCC system_header
#  endif
# endif

# include <libio/stdio.h>
# ifndef _ISOMAC

#  define _LIBC_STDIO_H 1
#  include <libio/libio.h>

/* Now define the internal interfaces.  */

/*  Some libc_hidden_ldbl_proto's do not map to a unique symbol when
    redirecting ldouble to _Float128 variants.  We can therefore safely
    directly alias them to their internal name.  */
# if __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1 && IS_IN (libc)
#  define stdio_hidden_ldbl_proto(p, f) \
  extern __typeof (p ## f) p ## f __asm (__ASMNAME ("___ieee128_" #f));
# elif __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
#  define stdio_hidden_ldbl_proto(p,f) __LDBL_REDIR1_DECL (p ## f, p ## f ## ieee128)
# else
#  define stdio_hidden_ldbl_proto(p,f) libc_hidden_proto (p ## f)
# endif

/* Set the error indicator on FP.  */
static inline void
fseterr_unlocked (FILE *fp)
{
  fp->_flags |= _IO_ERR_SEEN;
}

extern int __fcloseall (void) attribute_hidden;
extern int __snprintf (char *__restrict __s, size_t __maxlen,
		       const char *__restrict __format, ...)
     __attribute__ ((__format__ (__printf__, 3, 4)));
stdio_hidden_ldbl_proto (__, snprintf)

extern int __vfscanf (FILE *__restrict __s,
		      const char *__restrict __format,
		      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0)));
libc_hidden_proto (__vfscanf)
extern int __vscanf (const char *__restrict __format,
		     __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 1, 0)));
extern __ssize_t __getline (char **__lineptr, size_t *__n,
                            FILE *__stream) attribute_hidden;
extern int __vsscanf (const char *__restrict __s,
		      const char *__restrict __format,
		      __gnuc_va_list __arg)
     __attribute__ ((__format__ (__scanf__, 2, 0)));

extern int __sprintf_chk (char *, int, size_t, const char *, ...) __THROW;
extern int __snprintf_chk (char *, size_t, int, size_t, const char *, ...)
     __THROW;
extern int __vsprintf_chk (char *, int, size_t, const char *,
			   __gnuc_va_list) __THROW;
extern int __vsnprintf_chk (char *, size_t, int, size_t, const char *,
			    __gnuc_va_list) __THROW;
extern int __printf_chk (int, const char *, ...);
extern int __fprintf_chk (FILE *, int, const char *, ...);
extern int __vprintf_chk (int, const char *, __gnuc_va_list);
extern int __vfprintf_chk (FILE *, int, const char *, __gnuc_va_list);
extern char *__fgets_unlocked_chk (char *buf, size_t size, int n, FILE *fp);
extern char *__fgets_chk (char *buf, size_t size, int n, FILE *fp);
extern int __asprintf_chk (char **, int, const char *, ...) __THROW;
extern int __vasprintf_chk (char **, int, const char *, __gnuc_va_list) __THROW;
extern int __dprintf_chk (int, int, const char *, ...);
extern int __vdprintf_chk (int, int, const char *, __gnuc_va_list);
extern int __obstack_printf_chk (struct obstack *, int, const char *, ...)
     __THROW;
extern int __obstack_vprintf_chk (struct obstack *, int, const char *,
				  __gnuc_va_list) __THROW;

extern int __isoc99_fscanf (FILE *__restrict __stream,
			    const char *__restrict __format, ...) __wur;
extern int __isoc99_scanf (const char *__restrict __format, ...) __wur;
extern int __isoc99_sscanf (const char *__restrict __s,
			    const char *__restrict __format, ...) __THROW;
extern int __isoc99_vfscanf (FILE *__restrict __s,
			     const char *__restrict __format,
			     __gnuc_va_list __arg) __wur;
extern int __isoc99_vscanf (const char *__restrict __format,
			    __gnuc_va_list __arg) __wur;
extern int __isoc99_vsscanf (const char *__restrict __s,
			     const char *__restrict __format,
			     __gnuc_va_list __arg) __THROW;

libc_hidden_proto (__isoc99_sscanf)
libc_hidden_proto (__isoc99_vsscanf)
libc_hidden_proto (__isoc99_vfscanf)

/* Internal uses of sscanf should call the C99-compliant version.
   Unfortunately, symbol redirection is not transitive, so the
   __REDIRECT in the public header does not link up with the above
   libc_hidden_proto.  Bridge the gap with a macro.  */
#  if !__GLIBC_USE (DEPRECATED_SCANF)
#   undef sscanf
#   define sscanf __isoc99_sscanf
#  endif

#  if __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1  && IS_IN (libc)
/* These are implemented as redirects to other public API.
   Therefore, the usual redirection fails to avoid PLT.  */
extern __typeof (__isoc99_sscanf) ___ieee128_isoc99_sscanf __THROW;
extern __typeof (__isoc99_vsscanf) ___ieee128_isoc99_vsscanf __THROW;
extern __typeof (__isoc99_vfscanf) ___ieee128_isoc99_vfscanf __THROW;
libc_hidden_proto (___ieee128_isoc99_sscanf)
libc_hidden_proto (___ieee128_isoc99_vsscanf)
libc_hidden_proto (___ieee128_isoc99_vfscanf)
#define __isoc99_sscanf ___ieee128_isoc99_sscanf
#define __isoc99_vsscanf ___ieee128_isoc99_vsscanf
#define __isoc99_vfscanf ___ieee128_isoc99_vfscanf
#  endif

/* Prototypes for compatibility functions.  */
extern FILE *__new_tmpfile (void);
extern FILE *__old_tmpfile (void);

#  define __need_size_t
#  include <stddef.h>

#  include <bits/types/wint_t.h>

/* Generate a unique file name (and possibly open it).  */
extern int __path_search (char *__tmpl, size_t __tmpl_len,
			  const char *__dir, const char *__pfx,
			  int __try_tempdir) attribute_hidden;

extern int __gen_tempname (char *__tmpl, int __suffixlen, int __flags,
			   int __kind) attribute_hidden;
/* The __kind argument to __gen_tempname may be one of: */
#  define __GT_FILE	0	/* create a file */
#  define __GT_DIR	1	/* create a directory */
#  define __GT_NOCREATE	2	/* just find a name not currently in use */

enum __libc_message_action
{
  do_message	= 0,		/* Print message.  */
  do_abort	= 1 << 0,	/* Abort.  */
};

/* Print out MESSAGE (which should end with a newline) on the error output
   and abort.  */
extern void __libc_fatal (const char *__message)
     __attribute__ ((__noreturn__));
extern void __libc_message (enum __libc_message_action action,
			    const char *__fnt, ...) attribute_hidden;
extern void __fortify_fail (const char *msg) __attribute__ ((__noreturn__));
libc_hidden_proto (__fortify_fail)

/* Acquire ownership of STREAM.  */
extern void __flockfile (FILE *__stream) attribute_hidden;

/* Relinquish the ownership granted for STREAM.  */
extern void __funlockfile (FILE *__stream) attribute_hidden;

/* Try to acquire ownership of STREAM but do not block if it is not
   possible.  */
extern int __ftrylockfile (FILE *__stream);

extern int __getc_unlocked (FILE *__fp) attribute_hidden;
extern wint_t __getwc_unlocked (FILE *__fp);

extern int __fxprintf (FILE *__fp, const char *__fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3))) attribute_hidden;
extern int __fxprintf_nocancel (FILE *__fp, const char *__fmt, ...)
     __attribute__ ((__format__ (__printf__, 2, 3))) attribute_hidden;
int __vfxprintf (FILE *__fp, const char *__fmt, __gnuc_va_list,
		 unsigned int)
  attribute_hidden;

extern const char *const _sys_errlist_internal[] attribute_hidden;
extern const char *__get_errlist (int) attribute_hidden;
extern const char *__get_errname (int) attribute_hidden;

libc_hidden_ldbl_proto (__asprintf)

#  if IS_IN (libc)
extern FILE *_IO_new_fopen (const char*, const char*);
#   define fopen(fname, mode) _IO_new_fopen (fname, mode)
extern FILE *_IO_new_fdopen (int, const char*);
#   define fdopen(fd, mode) _IO_new_fdopen (fd, mode)
extern int _IO_new_fclose (FILE*);
#   define fclose(fp) _IO_new_fclose (fp)
extern int _IO_fputs (const char*, FILE*);
libc_hidden_proto (_IO_fputs)
/* The compiler may optimize calls to fprintf into calls to fputs.
   Use libc_hidden_proto to ensure that those calls, not redirected by
   the fputs macro, also do not go through the PLT.  */
libc_hidden_proto (fputs)
#   define fputs(str, fp) _IO_fputs (str, fp)
extern int _IO_new_fsetpos (FILE *, const __fpos_t *);
#   define fsetpos(fp, posp) _IO_new_fsetpos (fp, posp)
extern int _IO_new_fgetpos (FILE *, __fpos_t *);
#   define fgetpos(fp, posp) _IO_new_fgetpos (fp, posp)
#  endif

extern __typeof (dprintf) __dprintf
     __attribute__ ((__format__ (__printf__, 2, 3)));
stdio_hidden_ldbl_proto (__, dprintf)
libc_hidden_ldbl_proto (dprintf)
libc_hidden_ldbl_proto (fprintf)
libc_hidden_ldbl_proto (vfprintf)
libc_hidden_ldbl_proto (sprintf)
libc_hidden_proto (ungetc)
libc_hidden_proto (__getdelim)
libc_hidden_proto (fwrite)
libc_hidden_proto (perror)
libc_hidden_proto (remove)
libc_hidden_proto (rewind)
libc_hidden_proto (fileno)
extern __typeof (fileno) __fileno;
libc_hidden_proto (__fileno)
libc_hidden_proto (fwrite)
libc_hidden_proto (fseek)
extern __typeof (ftello) __ftello;
libc_hidden_proto (__ftello)
extern __typeof (fseeko64) __fseeko64;
libc_hidden_proto (__fseeko64)
extern __typeof (ftello64) __ftello64;
libc_hidden_proto (__ftello64)
libc_hidden_proto (fflush)
libc_hidden_proto (fflush_unlocked)
extern __typeof (fflush_unlocked) __fflush_unlocked;
libc_hidden_proto (__fflush_unlocked)
extern __typeof (fread_unlocked) __fread_unlocked;
libc_hidden_proto (__fread_unlocked)
libc_hidden_proto (fwrite_unlocked)
libc_hidden_proto (fgets_unlocked)
extern __typeof (fgets_unlocked) __fgets_unlocked;
libc_hidden_proto (__fgets_unlocked)
libc_hidden_proto (fputs_unlocked)
extern __typeof (fputs_unlocked) __fputs_unlocked;
libc_hidden_proto (__fputs_unlocked)
libc_hidden_proto (feof_unlocked)
extern __typeof (feof_unlocked) __feof_unlocked attribute_hidden;
libc_hidden_proto (ferror_unlocked)
extern __typeof (ferror_unlocked) __ferror_unlocked attribute_hidden;
libc_hidden_proto (getc_unlocked)
libc_hidden_proto (fputc_unlocked)
libc_hidden_proto (putc_unlocked)
extern __typeof (putc_unlocked) __putc_unlocked attribute_hidden;
libc_hidden_proto (fmemopen)
/* The prototype needs repeating instead of using __typeof to use
   __THROW in C++ tests.  */
extern FILE *__open_memstream (char **, size_t *) __THROW __wur;
libc_hidden_proto (__open_memstream)
libc_hidden_proto (__libc_fatal)
rtld_hidden_proto (__libc_fatal)
libc_hidden_proto (__vsprintf_chk)

extern FILE * __fmemopen (void *buf, size_t len, const char *mode);
libc_hidden_proto (__fmemopen)

extern int __gen_tempfd (int flags);
libc_hidden_proto (__gen_tempfd)

#  ifdef __USE_EXTERN_INLINES
__extern_inline int
__NTH (__feof_unlocked (FILE *__stream))
{
  return __feof_unlocked_body (__stream);
}

__extern_inline int
__NTH (__ferror_unlocked (FILE *__stream))
{
  return __ferror_unlocked_body (__stream);
}

__extern_inline int
__getc_unlocked (FILE *__fp)
{
  return __getc_unlocked_body (__fp);
}

__extern_inline int
__putc_unlocked (int __c, FILE *__stream)
{
  return __putc_unlocked_body (__c, __stream);
}
#  endif

extern __typeof (renameat) __renameat;
libc_hidden_proto (__renameat)
extern __typeof (renameat2) __renameat2;
libc_hidden_proto (__renameat2)

# endif /* not _ISOMAC */
#endif /* stdio.h */
