// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.unix.Stream -analyzer-store region -verify %s

typedef __typeof__(sizeof(int)) size_t;
typedef struct _IO_FILE FILE;
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */
extern FILE *fopen(const char *path, const char *mode);
extern FILE *tmpfile(void);
extern int fclose(FILE *fp);
extern size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern int fseek (FILE *__stream, long int __off, int __whence);
extern long int ftell (FILE *__stream);
extern void rewind (FILE *__stream);

void f1(void) {
  FILE *p = fopen("foo", "r");
  char buf[1024];
  fread(buf, 1, 1, p); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f2(void) {
  FILE *p = fopen("foo", "r");
  fseek(p, 1, SEEK_SET); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f3(void) {
  FILE *p = fopen("foo", "r");
  ftell(p); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f4(void) {
  FILE *p = fopen("foo", "r");
  rewind(p); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f5(void) {
  FILE *p = fopen("foo", "r");
  if (!p)
    return;
  fseek(p, 1, SEEK_SET); // no-warning
  fseek(p, 1, 3); // expected-warning {{The whence argument to fseek() should be SEEK_SET, SEEK_END, or SEEK_CUR}}
  fclose(p);
}

void f6(void) {
  FILE *p = fopen("foo", "r");
  fclose(p); 
  fclose(p); // expected-warning {{Try to close a file Descriptor already closed. Cause undefined behaviour}}
}

void f7(void) {
  FILE *p = tmpfile();
  ftell(p); // expected-warning {{Stream pointer might be NULL}}
  fclose(p);
}

void f8(int c) {
  FILE *p = fopen("foo.c", "r");
  if(c)
    return; // expected-warning {{Opened File never closed. Potential Resource leak}}
  fclose(p);
}

FILE *f9(void) {
  FILE *p = fopen("foo.c", "r");
  if (p)
    return p; // no-warning
  else
    return 0;
}

void pr7831(FILE *fp) {
  fclose(fp); // no-warning
}

// PR 8081 - null pointer crash when 'whence' is not an integer constant
void pr8081(FILE *stream, long offset, int whence) {
  fseek(stream, offset, whence);
}

