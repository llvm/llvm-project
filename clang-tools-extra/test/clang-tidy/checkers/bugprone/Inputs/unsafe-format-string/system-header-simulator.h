#pragma clang system_header

#ifdef __cplusplus
#define restrict /*restrict*/
#endif

#ifndef __cplusplus
typedef __WCHAR_TYPE__ wchar_t;
#endif

typedef __typeof(sizeof(int)) size_t;
typedef long long __int64_t;
typedef __int64_t __darwin_off_t;
typedef __darwin_off_t fpos_t;
typedef int off_t;
typedef long ssize_t;

typedef struct _FILE FILE;

extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

typedef __builtin_va_list va_list;
#define va_start(ap, param) __builtin_va_start(ap, param)
#define va_end(ap)          __builtin_va_end(ap)
#define va_arg(ap, type)    __builtin_va_arg(ap, type)
#define va_copy(dst, src)   __builtin_va_copy(dst, src)


#ifdef __cplusplus
namespace std {
#endif
extern int fscanf ( FILE *restrict stream, const char *restrict format, ... );
extern int scanf ( const char *restrict format, ... );
extern int sscanf ( const char *restrict s, const char *restrict format, ...);
extern int vscanf( const char *restrict format, va_list vlist );
extern int vfscanf ( FILE *restrict stream, const char *restrict format, va_list arg );

extern int vsscanf( const char *restrict buffer, const char *restrict format, va_list vlist );
extern int vwscanf( const wchar_t* format, va_list vlist );
extern int vfwscanf( FILE* stream, const wchar_t* format, va_list vlist );
extern int vswscanf( const wchar_t* buffer, const wchar_t* format, va_list vlist );
extern int swscanf (const wchar_t* ws, const wchar_t* format, ...);
extern int wscanf( const wchar_t *format, ... );
extern int fwscanf( FILE *stream, const wchar_t *format, ... );

extern int printf( const char*          format, ... );
extern int sprintf( char* buffer, const char* format, ... );
extern int vsprintf (char * s, const char * format, va_list arg );
extern int vsnprintf (char * s, size_t n, const char * format, va_list arg );
extern int fprintf( FILE*          stream, const char*          format, ... );
extern int snprintf( char* restrict buffer, size_t bufsz,
              const char* restrict format, ... );
#ifdef __cplusplus
} //namespace std {
#endif