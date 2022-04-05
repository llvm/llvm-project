#ifndef BOLT_TEST_STUB_H
#define BOLT_TEST_STUB_H

void *memcpy(void *dest, const void *src, unsigned long n);
void *memset(void *dest, int c, unsigned long n);
int printf(const char *format, ...);
void exit(int status);

#endif
