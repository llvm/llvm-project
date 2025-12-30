#ifndef SETJMP_H
#define SETJMP_H

typedef struct {
  int x[42];
} jmp_buf;

 __attribute__((noreturn))
void longjmp(jmp_buf, int);

#endif
