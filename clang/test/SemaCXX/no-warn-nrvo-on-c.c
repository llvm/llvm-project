// RUN: %clang -std=c23 -Wnrvo -Xclang -verify %s
// expected-no-diagnostics

#include <stdlib.h>

#define SIZE 20

typedef struct String_s {
    char*  buf;
    size_t len;
} String;


void clean(String* s) {
    free(s->buf);
}

String randomString() {
    String s = {};

    s.buf = malloc(SIZE);
    s.len = SIZE;

    if (!s.buf) {
        goto fail;
    }

    return s;

fail:
    clean(&s);
    return (String){};
}

int main(int argc, char** argv)
{
    String s= randomString();
    clean(&s);

    return 0;
}
