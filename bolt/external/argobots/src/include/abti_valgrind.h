/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_VALGRIND_H_INCLUDED
#define ABTI_VALGRIND_H_INCLUDED

/* Valgrind support */
#ifdef HAVE_VALGRIND_SUPPORT

#include <valgrind/valgrind.h>

void ABTI_valgrind_register_stack(const void *p_stack, size_t size);
void ABTI_valgrind_unregister_stack(const void *p_stack);
#define ABTI_VALGRIND_REGISTER_STACK(p_stack, size)                            \
    do {                                                                       \
        if (!RUNNING_ON_VALGRIND)                                              \
            break;                                                             \
        ABTI_valgrind_register_stack(p_stack, size);                           \
    } while (0)

#define ABTI_VALGRIND_UNREGISTER_STACK(p_stack)                                \
    do {                                                                       \
        if (!RUNNING_ON_VALGRIND)                                              \
            break;                                                             \
        ABTI_valgrind_unregister_stack(p_stack);                               \
    } while (0)

#else

#define ABTI_VALGRIND_REGISTER_STACK(p_stack, size)                            \
    do {                                                                       \
    } while (0)
#define ABTI_VALGRIND_UNREGISTER_STACK(p_stack)                                \
    do {                                                                       \
    } while (0)

#endif /* HAVE_VALGRIND_SUPPORT */

#endif /* ABTI_VALGRIND_H_INCLUDED */
