// RUN: %clang_cc1 -Wstrlcpy-strlcat-size -verify -fsyntax-only %s

typedef __SIZE_TYPE__ size_t;
size_t strlcpy (char * restrict dst, const char * restrict src, size_t size);
size_t strlcat (char * restrict dst, const char * restrict src, size_t size);
size_t strlen (const char *s);

char s1[100];
char s2[200];
char * s3;

struct {
  char f1[100];
  char f2[100][3];
} s4, **s5;

int x;

void f(void)
{
  strlcpy(s1, s2, sizeof(s1)); // no warning
  strlcpy(s1, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s1, s3, strlen(s3)+1); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcat(s2, s3, sizeof(s3)); // expected-warning {{size argument in 'strlcat' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s4.f1, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy((*s5)->f2[x], s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
  strlcpy(s1+3, s2, sizeof(s2)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}}
}

// Don't issue FIXIT for flexible arrays.
struct S {
  int y; 
  char x[];
};

void flexible_arrays(struct S *s) {
  char str[] = "hi";
  strlcpy(s->x, str, sizeof(str));  // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}}
}

// Don't issue FIXIT for destinations of size 1.
void size_1(void) {
  char z[1];
  char str[] = "hi";

  strlcpy(z, str, sizeof(str));  // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}}
}

// Support VLAs.
void vlas(int size) {
  char z[size];
  char str[] = "hi";

  strlcpy(z, str, sizeof(str)); // expected-warning {{size argument in 'strlcpy' call appears to be size of the source; expected the size of the destination}} expected-note {{change size argument to be the size of the destination}}
}
