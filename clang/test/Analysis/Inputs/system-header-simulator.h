#pragma clang system_header

typedef struct _FILE FILE;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
// Include a variant of standard streams that occur in the pre-processed file.
extern FILE *__stdinp;
extern FILE *__stdoutp;
extern FILE *__stderrp;


int fscanf(FILE *restrict, const char *restrict, ...);

// Note, on some platforms errno macro gets replaced with a function call.
extern int errno;

typedef __typeof(sizeof(int)) size_t;

size_t strlen(const char *);

char *strcpy(char *restrict, const char *restrict);

typedef unsigned long __darwin_pthread_key_t;
typedef __darwin_pthread_key_t pthread_key_t;
int pthread_setspecific(pthread_key_t, const void *);

typedef long long __int64_t;
typedef __int64_t __darwin_off_t;
typedef __darwin_off_t fpos_t;

void setbuf(FILE * restrict, char * restrict);
int setvbuf(FILE * restrict, char * restrict, int, size_t);

FILE *funopen(const void *,
                 int (*)(void *, char *, int),
                 int (*)(void *, const char *, int),
                 fpos_t (*)(void *, fpos_t, int),
                 int (*)(void *));

int sqlite3_bind_text_my(int, const char*, int n, void(*)(void*));

typedef void (*freeCallback) (void*);
typedef struct {
  int i;
  freeCallback fc;
} StWithCallback;

int dealocateMemWhenDoneByVal(void*, StWithCallback);
int dealocateMemWhenDoneByRef(StWithCallback*, const void*);

typedef struct CGContext *CGContextRef;
CGContextRef CGBitmapContextCreate(void *data/*, size_t width, size_t height,
                                   size_t bitsPerComponent, size_t bytesPerRow,
                                   CGColorSpaceRef space,
                                   CGBitmapInfo bitmapInfo*/);
void *CGBitmapContextGetData(CGContextRef context);

// Include xpc.
typedef struct _xpc_connection_s * xpc_connection_t;
typedef void (*xpc_finalizer_t)(void *value);
void xpc_connection_set_context(xpc_connection_t connection, void *context);
void xpc_connection_set_finalizer_f(xpc_connection_t connection, xpc_finalizer_t finalizer);
void xpc_connection_resume(xpc_connection_t connection);
