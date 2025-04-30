#ifndef _IOLIBIO_H
#define _IOLIBIO_H 1

#include <stdio.h>
#include <libio/libio.h>

/* Alternative names for many of the stdio.h functions, used
   internally and exposed for backward compatibility's sake.  */

extern int _IO_fclose (FILE*);
extern int _IO_new_fclose (FILE*);
extern int _IO_old_fclose (FILE*);
extern FILE *_IO_fdopen (int, const char*) __THROW;
libc_hidden_proto (_IO_fdopen)
extern FILE *_IO_old_fdopen (int, const char*) __THROW;
extern FILE *_IO_new_fdopen (int, const char*) __THROW;
extern int _IO_fflush (FILE*);
libc_hidden_proto (_IO_fflush)
extern int _IO_fgetpos (FILE*, __fpos_t*);
extern int _IO_fgetpos64 (FILE*, __fpos64_t*);
extern char* _IO_fgets (char*, int, FILE*);
extern FILE *_IO_fopen (const char*, const char*);
extern FILE *_IO_old_fopen (const char*, const char*);
extern FILE *_IO_new_fopen (const char*, const char*);
extern FILE *_IO_fopen64 (const char*, const char*);
extern FILE *__fopen_internal (const char*, const char*, int)
	attribute_hidden;
extern FILE *__fopen_maybe_mmap (FILE *) __THROW attribute_hidden;
extern int _IO_fprintf (FILE*, const char*, ...);
extern int _IO_fputs (const char*, FILE*);
libc_hidden_proto (_IO_fputs)
extern int _IO_fsetpos (FILE*, const __fpos_t *);
extern int _IO_fsetpos64 (FILE*, const __fpos64_t *);
extern long int _IO_ftell (FILE*);
libc_hidden_proto (_IO_ftell)
extern size_t _IO_fread (void*, size_t, size_t, FILE*);
libc_hidden_proto (_IO_fread)
extern size_t _IO_fwrite (const void*, size_t, size_t, FILE*);
libc_hidden_proto (_IO_fwrite)
extern char* _IO_gets (char*);
extern void _IO_perror (const char*) __THROW;
extern int _IO_printf (const char*, ...);
extern int _IO_puts (const char*);
libc_hidden_proto (_IO_puts)
extern int _IO_scanf (const char*, ...);
extern void _IO_setbuffer (FILE *, char*, size_t) __THROW;
libc_hidden_proto (_IO_setbuffer)
extern int _IO_setvbuf (FILE*, char*, int, size_t) __THROW;
libc_hidden_proto (_IO_setvbuf)
extern int _IO_sscanf (const char*, const char*, ...) __THROW;
extern int _IO_sprintf (char *, const char*, ...) __THROW;
extern int _IO_ungetc (int, FILE*) __THROW;
extern int _IO_vsscanf (const char *, const char *, __gnuc_va_list) __THROW;

#define _IO_clearerr(FP) ((FP)->_flags &= ~(_IO_ERR_SEEN|_IO_EOF_SEEN))
#define _IO_fseek(__fp, __offset, __whence) \
  (_IO_seekoff_unlocked (__fp, __offset, __whence, _IOS_INPUT|_IOS_OUTPUT) \
   == _IO_pos_BAD ? EOF : 0)
#define _IO_rewind(FILE) \
  (void) _IO_seekoff_unlocked (FILE, 0, 0, _IOS_INPUT|_IOS_OUTPUT)
#define _IO_freopen(FILENAME, MODE, FP) \
  (_IO_file_close_it (FP), \
   _IO_file_fopen (FP, FILENAME, MODE, 1))
#define _IO_old_freopen(FILENAME, MODE, FP) \
  (_IO_old_file_close_it (FP), _IO_old_file_fopen(FP, FILENAME, MODE))
#define _IO_freopen64(FILENAME, MODE, FP) \
  (_IO_file_close_it (FP), \
   _IO_file_fopen (FP, FILENAME, MODE, 0))
#define _IO_fileno(FP) ((FP)->_fileno)
extern FILE* _IO_popen (const char*, const char*) __THROW;
extern FILE* _IO_new_popen (const char*, const char*) __THROW;
extern FILE* _IO_old_popen (const char*, const char*) __THROW;
extern int __new_pclose (FILE *) __THROW;
extern int __old_pclose (FILE *) __THROW;
#define _IO_pclose _IO_fclose
#define _IO_setbuf(_FP, _BUF) _IO_setbuffer (_FP, _BUF, BUFSIZ)
#define _IO_setlinebuf(_FP) _IO_setvbuf (_FP, NULL, 1, 0)

FILE *__new_freopen (const char *, const char *, FILE *) __THROW;
FILE *__old_freopen (const char *, const char *, FILE *) __THROW;

#endif /* iolibio.h.  */
