#pragma once
typedef __SIZE_TYPE__ size_t;

void *malloc(size_t);
void free(void*);

#ifndef __cplusplus
extern int abs(int __x) __attribute__((__const__));
#endif
