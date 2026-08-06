// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

// all cases should be ignored in this file

extern int *errno;

void foo(int errno) {}

int fooo()
{
    int errno = 0;
    return errno;
}
