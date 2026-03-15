#pragma once
typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void free(void*);

#ifndef __cplusplus
extern int abs(int __x) __attribute__((__const__));
extern long labs(long __x) __attribute__((__const__));
extern long long llabs(long long __x) __attribute__((__const__));
#endif

void free(void* ptr);
void* malloc(size_t size);
