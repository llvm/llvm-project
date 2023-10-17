typedef __SIZE_TYPE__ size_t;
#define __SSIZE_TYPE__                                                         \
  __typeof__(_Generic((__SIZE_TYPE__)0,                                        \
                      unsigned long long int : (long long int)0,               \
                      unsigned long int : (long int)0,                         \
                      unsigned int : (int)0,                                   \
                      unsigned short : (short)0))
typedef __SSIZE_TYPE__ ssize_t;
typedef struct {
  int x;
} FILE;

// do not use the default values for these constants to verify that this
// definition is found
#define EOF (-2)
#define AT_FDCWD (-101)

#ifdef __cplusplus
#define restrict /*restrict*/
#endif

int isascii(int);
int islower(int);
int isalpha(int);
int isalnum(int);
int isblank(int);
int ispunct(int);
int isupper(int);
int isgraph(int);
int isprint(int);
int isdigit(int);
int isspace(int);
int isxdigit(int);
int toupper(int);
int tolower(int);
int toascii(int);

int getc(FILE *);
int fgetc(FILE *);
int getchar(void);
size_t fread(void *restrict, size_t, size_t, FILE *restrict);
size_t fwrite(const void *restrict, size_t, size_t, FILE *restrict);
ssize_t read(int, void *, size_t);
ssize_t write(int, const void *, size_t);
ssize_t getline(char **restrict, size_t *restrict, FILE *restrict);
ssize_t getdelim(char **restrict, size_t *restrict, int, FILE *restrict);
char *getenv(const char *);
