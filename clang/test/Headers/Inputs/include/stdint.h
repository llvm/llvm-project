#ifndef STDINT_H
#define STDINT_H

#ifdef __INT8_TYPE__
typedef __INT8_TYPE__ int8_t;
#endif
typedef unsigned char uint8_t;

#ifdef __INT16_TYPE__
typedef __INT16_TYPE__ int16_t;
typedef unsigned __INT16_TYPE__ uint16_t;
#endif

#ifdef __INT32_TYPE__
typedef __INT32_TYPE__ int32_t;
typedef unsigned __INT32_TYPE__ uint32_t;
#endif

#ifdef __INT64_TYPE__
typedef __INT64_TYPE__ int64_t;
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
