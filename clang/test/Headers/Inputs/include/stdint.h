#ifndef STDINT_H
#define STDINT_H

#ifdef __INT32_TYPE__
typedef unsigned __INT32_TYPE__ uint32_t;
#endif

#ifdef __INT64_TYPE__
typedef unsigned __INT64_TYPE__ uint64_t;
#endif

#ifdef __INTPTR_TYPE__
typedef __INTPTR_TYPE__ intptr_t;
typedef unsigned __INTPTR_TYPE__ uintptr_t;
#else
#error Every target should have __INTPTR_TYPE__
#endif

#ifdef __INTPTR_MAX__
#define  INTPTR_MAX    __INTPTR_MAX__
#endif

#ifdef __UINTPTR_MAX__
#define UINTPTR_MAX   __UINTPTR_MAX__
#endif

#endif /* STDINT_H */
