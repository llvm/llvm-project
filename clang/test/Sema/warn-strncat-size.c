// RUN: %clang_cc1 -Wstrncat-size -verify -fsyntax-only %s
// RUN: %clang_cc1 -DUSE_BUILTINS -Wstrncat-size -verify -fsyntax-only %s
// RUN: %clang_cc1 -fsyntax-only -Wstrncat-size -fixit -x c %s
// RUN: %clang_cc1 -DUSE_BUILTINS -fsyntax-only -Wstrncat-size -fixit -x c %s

typedef __SIZE_TYPE__ size_t;
size_t strlen (const char *s);

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else
# define BUILTIN(f) f
#endif

#define strncat BUILTIN(strncat)
char *strncat(char *restrict s1, const char *restrict s2, size_t n);

struct {
  char f1[100];
  char f2[100][3];
} s4, **s5;

char s1[100];
char s2[200];
int x;

void test(char *src) {
  char dest[10];

  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - strlen(dest) - 1); // no-warning
  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - 1); // no-warning - the code might assume that dest is empty

  strncat(dest, src, sizeof(src)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}

  strncat(dest, src, sizeof(src) - 1); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}

  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest)); // expected-warning{{the value of the size argument in 'strncat' is too large, might lead to a buffer overflow}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}

  strncat(dest, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", sizeof(dest) - strlen(dest)); // expected-warning{{the value of the size argument in 'strncat' is too large, might lead to a buffer overflow}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}

  strncat((*s5)->f2[x], s2, sizeof(s2)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}
  strncat(s1+3, s2, sizeof(s2)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-warning {{strncat' size argument is too large; destination buffer has size 97, but size argument is 200}}
  strncat(s4.f1, s2, sizeof(s2)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}
}

// Don't issue FIXIT for flexible arrays.
struct S {
  int y;
  char x[];
};

void flexible_arrays(struct S *s) {
  char str[] = "hi";
  strncat(s->x, str,  sizeof(str)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}}
}

// Don't issue FIXIT for destinations of size 1.
void size_1(void) {
  char z[1];
  char str[] = "hi";

  strncat(z, str, sizeof(z)); // expected-warning{{the value of the size argument to 'strncat' is wrong}}
}

// Support VLAs.
void vlas(int size) {
  char z[size];
  char str[] = "hi";

  strncat(z, str, sizeof(str)); // expected-warning {{size argument in 'strncat' call appears to be size of the source}} expected-note {{change the argument to be the free space in the destination buffer minus the terminating null byte}}
}

// Non-array type gets a different error message.
void f(char* s, char* d) {
  strncat(d, s, sizeof(d)); // expected-warning {{the value of the size argument to 'strncat' is wrong}}
}
