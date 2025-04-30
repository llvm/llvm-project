/* Type-safe arrays which grow dynamically.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* Pre-processor macros which act as parameters:

   DYNARRAY_STRUCT
      The struct tag of dynamic array to be defined.
   DYNARRAY_ELEMENT
      The type name of the element type.  Elements are copied
      as if by memcpy, and can change address as the dynamic
      array grows.
   DYNARRAY_PREFIX
      The prefix of the functions which are defined.

   The following parameters are optional:

   DYNARRAY_ELEMENT_FREE
      DYNARRAY_ELEMENT_FREE (E) is evaluated to deallocate the
      contents of elements. E is of type  DYNARRAY_ELEMENT *.
   DYNARRAY_ELEMENT_INIT
      DYNARRAY_ELEMENT_INIT (E) is evaluated to initialize a new
      element.  E is of type  DYNARRAY_ELEMENT *.
      If DYNARRAY_ELEMENT_FREE but not DYNARRAY_ELEMENT_INIT is
      defined, new elements are automatically zero-initialized.
      Otherwise, new elements have undefined contents.
   DYNARRAY_INITIAL_SIZE
      The size of the statically allocated array (default:
      at least 2, more elements if they fit into 128 bytes).
      Must be a preprocessor constant.  If DYNARRAY_INITIAL_SIZE is 0,
      there is no statically allocated array at, and all non-empty
      arrays are heap-allocated.
   DYNARRAY_FINAL_TYPE
      The name of the type which holds the final array.  If not
      defined, is PREFIX##finalize not provided.  DYNARRAY_FINAL_TYPE
      must be a struct type, with members of type DYNARRAY_ELEMENT and
      size_t at the start (in this order).

   These macros are undefined after this header file has been
   included.

   The following types are provided (their members are private to the
   dynarray implementation):

     struct DYNARRAY_STRUCT

   The following functions are provided:

     void DYNARRAY_PREFIX##init (struct DYNARRAY_STRUCT *);
     void DYNARRAY_PREFIX##free (struct DYNARRAY_STRUCT *);
     bool DYNARRAY_PREFIX##has_failed (const struct DYNARRAY_STRUCT *);
     void DYNARRAY_PREFIX##mark_failed (struct DYNARRAY_STRUCT *);
     size_t DYNARRAY_PREFIX##size (const struct DYNARRAY_STRUCT *);
     DYNARRAY_ELEMENT *DYNARRAY_PREFIX##begin (const struct DYNARRAY_STRUCT *);
     DYNARRAY_ELEMENT *DYNARRAY_PREFIX##end (const struct DYNARRAY_STRUCT *);
     DYNARRAY_ELEMENT *DYNARRAY_PREFIX##at (struct DYNARRAY_STRUCT *, size_t);
     void DYNARRAY_PREFIX##add (struct DYNARRAY_STRUCT *, DYNARRAY_ELEMENT);
     DYNARRAY_ELEMENT *DYNARRAY_PREFIX##emplace (struct DYNARRAY_STRUCT *);
     bool DYNARRAY_PREFIX##resize (struct DYNARRAY_STRUCT *, size_t);
     void DYNARRAY_PREFIX##remove_last (struct DYNARRAY_STRUCT *);
     void DYNARRAY_PREFIX##clear (struct DYNARRAY_STRUCT *);

   The following functions are provided are provided if the
   prerequisites are met:

     bool DYNARRAY_PREFIX##finalize (struct DYNARRAY_STRUCT *,
                                     DYNARRAY_FINAL_TYPE *);
       (if DYNARRAY_FINAL_TYPE is defined)
     DYNARRAY_ELEMENT *DYNARRAY_PREFIX##finalize (struct DYNARRAY_STRUCT *,
                                                  size_t *);
       (if DYNARRAY_FINAL_TYPE is not defined)
*/

#include <malloc/dynarray.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef DYNARRAY_STRUCT
# error "DYNARRAY_STRUCT must be defined"
#endif

#ifndef DYNARRAY_ELEMENT
# error "DYNARRAY_ELEMENT must be defined"
#endif

#ifndef DYNARRAY_PREFIX
# error "DYNARRAY_PREFIX must be defined"
#endif

#ifdef DYNARRAY_INITIAL_SIZE
# if DYNARRAY_INITIAL_SIZE < 0
#  error "DYNARRAY_INITIAL_SIZE must be non-negative"
# endif
# if DYNARRAY_INITIAL_SIZE > 0
#  define DYNARRAY_HAVE_SCRATCH 1
# else
#  define DYNARRAY_HAVE_SCRATCH 0
# endif
#else
/* Provide a reasonable default which limits the size of
   DYNARRAY_STRUCT.  */
# define DYNARRAY_INITIAL_SIZE \
  (sizeof (DYNARRAY_ELEMENT) > 64 ? 2 : 128 / sizeof (DYNARRAY_ELEMENT))
# define DYNARRAY_HAVE_SCRATCH 1
#endif

/* Public type definitions.  */

/* All fields of this struct are private to the implementation.  */
struct DYNARRAY_STRUCT
{
  union
  {
    struct dynarray_header dynarray_abstract;
    struct
    {
      /* These fields must match struct dynarray_header.  */
      size_t used;
      size_t allocated;
      DYNARRAY_ELEMENT *array;
    } dynarray_header;
  } u;

#if DYNARRAY_HAVE_SCRATCH
  /* Initial inline allocation.  */
  DYNARRAY_ELEMENT scratch[DYNARRAY_INITIAL_SIZE];
#endif
};

/* Internal use only: Helper macros.  */

/* Ensure macro-expansion of DYNARRAY_PREFIX.  */
#define DYNARRAY_CONCAT0(prefix, name) prefix##name
#define DYNARRAY_CONCAT1(prefix, name) DYNARRAY_CONCAT0(prefix, name)
#define DYNARRAY_NAME(name) DYNARRAY_CONCAT1(DYNARRAY_PREFIX, name)

/* Use DYNARRAY_FREE instead of DYNARRAY_NAME (free),
   so that Gnulib does not change 'free' to 'rpl_free'.  */
#define DYNARRAY_FREE DYNARRAY_CONCAT1 (DYNARRAY_NAME (f), ree)

/* Address of the scratch buffer if any.  */
#if DYNARRAY_HAVE_SCRATCH
# define DYNARRAY_SCRATCH(list) (list)->scratch
#else
# define DYNARRAY_SCRATCH(list) NULL
#endif

/* Internal use only: Helper functions.  */

/* Internal function.  Call DYNARRAY_ELEMENT_FREE with the array
   elements.  Name mangling needed due to the DYNARRAY_ELEMENT_FREE
   macro expansion.  */
static inline void
DYNARRAY_NAME (free__elements__) (DYNARRAY_ELEMENT *__dynarray_array,
                                  size_t __dynarray_used)
{
#ifdef DYNARRAY_ELEMENT_FREE
  for (size_t __dynarray_i = 0; __dynarray_i < __dynarray_used; ++__dynarray_i)
    DYNARRAY_ELEMENT_FREE (&__dynarray_array[__dynarray_i]);
#endif /* DYNARRAY_ELEMENT_FREE */
}

/* Internal function.  Free the non-scratch array allocation.  */
static inline void
DYNARRAY_NAME (free__array__) (struct DYNARRAY_STRUCT *list)
{
#if DYNARRAY_HAVE_SCRATCH
  if (list->u.dynarray_header.array != list->scratch)
    free (list->u.dynarray_header.array);
#else
  free (list->u.dynarray_header.array);
#endif
}

/* Public functions.  */

/* Initialize a dynamic array object.  This must be called before any
   use of the object.  */
__nonnull ((1))
static void
DYNARRAY_NAME (init) (struct DYNARRAY_STRUCT *list)
{
  list->u.dynarray_header.used = 0;
  list->u.dynarray_header.allocated = DYNARRAY_INITIAL_SIZE;
  list->u.dynarray_header.array = DYNARRAY_SCRATCH (list);
}

/* Deallocate the dynamic array and its elements.  */
__attribute_maybe_unused__ __nonnull ((1))
static void
DYNARRAY_FREE (struct DYNARRAY_STRUCT *list)
{
  DYNARRAY_NAME (free__elements__)
    (list->u.dynarray_header.array, list->u.dynarray_header.used);
  DYNARRAY_NAME (free__array__) (list);
  DYNARRAY_NAME (init) (list);
}

/* Return true if the dynamic array is in an error state.  */
__nonnull ((1))
static inline bool
DYNARRAY_NAME (has_failed) (const struct DYNARRAY_STRUCT *list)
{
  return list->u.dynarray_header.allocated == __dynarray_error_marker ();
}

/* Mark the dynamic array as failed.  All elements are deallocated as
   a side effect.  */
__nonnull ((1))
static void
DYNARRAY_NAME (mark_failed) (struct DYNARRAY_STRUCT *list)
{
  DYNARRAY_NAME (free__elements__)
    (list->u.dynarray_header.array, list->u.dynarray_header.used);
  DYNARRAY_NAME (free__array__) (list);
  list->u.dynarray_header.array = DYNARRAY_SCRATCH (list);
  list->u.dynarray_header.used = 0;
  list->u.dynarray_header.allocated = __dynarray_error_marker ();
}

/* Return the number of elements which have been added to the dynamic
   array.  */
__nonnull ((1))
static inline size_t
DYNARRAY_NAME (size) (const struct DYNARRAY_STRUCT *list)
{
  return list->u.dynarray_header.used;
}

/* Return a pointer to the array element at INDEX.  Terminate the
   process if INDEX is out of bounds.  */
__nonnull ((1))
static inline DYNARRAY_ELEMENT *
DYNARRAY_NAME (at) (struct DYNARRAY_STRUCT *list, size_t index)
{
  if (__glibc_unlikely (index >= DYNARRAY_NAME (size) (list)))
    __libc_dynarray_at_failure (DYNARRAY_NAME (size) (list), index);
  return list->u.dynarray_header.array + index;
}

/* Return a pointer to the first array element, if any.  For a
   zero-length array, the pointer can be NULL even though the dynamic
   array has not entered the failure state.  */
__nonnull ((1))
static inline DYNARRAY_ELEMENT *
DYNARRAY_NAME (begin) (struct DYNARRAY_STRUCT *list)
{
  return list->u.dynarray_header.array;
}

/* Return a pointer one element past the last array element.  For a
   zero-length array, the pointer can be NULL even though the dynamic
   array has not entered the failure state.  */
__nonnull ((1))
static inline DYNARRAY_ELEMENT *
DYNARRAY_NAME (end) (struct DYNARRAY_STRUCT *list)
{
  return list->u.dynarray_header.array + list->u.dynarray_header.used;
}

/* Internal function.  Slow path for the add function below.  */
static void
DYNARRAY_NAME (add__) (struct DYNARRAY_STRUCT *list, DYNARRAY_ELEMENT item)
{
  if (__glibc_unlikely
      (!__libc_dynarray_emplace_enlarge (&list->u.dynarray_abstract,
                                         DYNARRAY_SCRATCH (list),
                                         sizeof (DYNARRAY_ELEMENT))))
    {
      DYNARRAY_NAME (mark_failed) (list);
      return;
    }

  /* Copy the new element and increase the array length.  */
  list->u.dynarray_header.array[list->u.dynarray_header.used++] = item;
}

/* Add ITEM at the end of the array, enlarging it by one element.
   Mark *LIST as failed if the dynamic array allocation size cannot be
   increased.  */
__nonnull ((1))
static inline void
DYNARRAY_NAME (add) (struct DYNARRAY_STRUCT *list, DYNARRAY_ELEMENT item)
{
  /* Do nothing in case of previous error.  */
  if (DYNARRAY_NAME (has_failed) (list))
    return;

  /* Enlarge the array if necessary.  */
  if (__glibc_unlikely (list->u.dynarray_header.used
                        == list->u.dynarray_header.allocated))
    {
      DYNARRAY_NAME (add__) (list, item);
      return;
    }

  /* Copy the new element and increase the array length.  */
  list->u.dynarray_header.array[list->u.dynarray_header.used++] = item;
}

/* Internal function.  Building block for the emplace functions below.
   Assumes space for one more element in *LIST.  */
static inline DYNARRAY_ELEMENT *
DYNARRAY_NAME (emplace__tail__) (struct DYNARRAY_STRUCT *list)
{
  DYNARRAY_ELEMENT *result
    = &list->u.dynarray_header.array[list->u.dynarray_header.used];
  ++list->u.dynarray_header.used;
#if defined (DYNARRAY_ELEMENT_INIT)
  DYNARRAY_ELEMENT_INIT (result);
#elif defined (DYNARRAY_ELEMENT_FREE)
  memset (result, 0, sizeof (*result));
#endif
  return result;
}

/* Internal function.  Slow path for the emplace function below.  */
static DYNARRAY_ELEMENT *
DYNARRAY_NAME (emplace__) (struct DYNARRAY_STRUCT *list)
{
  if (__glibc_unlikely
      (!__libc_dynarray_emplace_enlarge (&list->u.dynarray_abstract,
                                         DYNARRAY_SCRATCH (list),
                                         sizeof (DYNARRAY_ELEMENT))))
    {
      DYNARRAY_NAME (mark_failed) (list);
      return NULL;
    }
  return DYNARRAY_NAME (emplace__tail__) (list);
}

/* Allocate a place for a new element in *LIST and return a pointer to
   it.  The pointer can be NULL if the dynamic array cannot be
   enlarged due to a memory allocation failure.  */
__attribute_maybe_unused__ __attribute_warn_unused_result__ __nonnull ((1))
static
/* Avoid inlining with the larger initialization code.  */
#if !(defined (DYNARRAY_ELEMENT_INIT) || defined (DYNARRAY_ELEMENT_FREE))
inline
#endif
DYNARRAY_ELEMENT *
DYNARRAY_NAME (emplace) (struct DYNARRAY_STRUCT *list)
{
  /* Do nothing in case of previous error.  */
  if (DYNARRAY_NAME (has_failed) (list))
    return NULL;

  /* Enlarge the array if necessary.  */
  if (__glibc_unlikely (list->u.dynarray_header.used
                        == list->u.dynarray_header.allocated))
    return (DYNARRAY_NAME (emplace__) (list));
  return DYNARRAY_NAME (emplace__tail__) (list);
}

/* Change the size of *LIST to SIZE.  If SIZE is larger than the
   existing size, new elements are added (which can be initialized).
   Otherwise, the list is truncated, and elements are freed.  Return
   false on memory allocation failure (and mark *LIST as failed).  */
__attribute_maybe_unused__ __nonnull ((1))
static bool
DYNARRAY_NAME (resize) (struct DYNARRAY_STRUCT *list, size_t size)
{
  if (size > list->u.dynarray_header.used)
    {
      bool ok;
#if defined (DYNARRAY_ELEMENT_INIT)
      /* The new elements have to be initialized.  */
      size_t old_size = list->u.dynarray_header.used;
      ok = __libc_dynarray_resize (&list->u.dynarray_abstract,
                                   size, DYNARRAY_SCRATCH (list),
                                   sizeof (DYNARRAY_ELEMENT));
      if (ok)
        for (size_t i = old_size; i < size; ++i)
          {
            DYNARRAY_ELEMENT_INIT (&list->u.dynarray_header.array[i]);
          }
#elif defined (DYNARRAY_ELEMENT_FREE)
      /* Zero initialization is needed so that the elements can be
         safely freed.  */
      ok = __libc_dynarray_resize_clear
        (&list->u.dynarray_abstract, size,
         DYNARRAY_SCRATCH (list), sizeof (DYNARRAY_ELEMENT));
#else
      ok =  __libc_dynarray_resize (&list->u.dynarray_abstract,
                                    size, DYNARRAY_SCRATCH (list),
                                    sizeof (DYNARRAY_ELEMENT));
#endif
      if (__glibc_unlikely (!ok))
        DYNARRAY_NAME (mark_failed) (list);
      return ok;
    }
  else
    {
      /* The list has shrunk in size.  Free the removed elements.  */
      DYNARRAY_NAME (free__elements__)
        (list->u.dynarray_header.array + size,
         list->u.dynarray_header.used - size);
      list->u.dynarray_header.used = size;
      return true;
    }
}

/* Remove the last element of LIST if it is present.  */
__attribute_maybe_unused__ __nonnull ((1))
static void
DYNARRAY_NAME (remove_last) (struct DYNARRAY_STRUCT *list)
{
  /* used > 0 implies that the array is the non-failed state.  */
  if (list->u.dynarray_header.used > 0)
    {
      size_t new_length = list->u.dynarray_header.used - 1;
#ifdef DYNARRAY_ELEMENT_FREE
      DYNARRAY_ELEMENT_FREE (&list->u.dynarray_header.array[new_length]);
#endif
      list->u.dynarray_header.used = new_length;
    }
}

/* Remove all elements from the list.  The elements are freed, but the
   list itself is not.  */
__attribute_maybe_unused__ __nonnull ((1))
static void
DYNARRAY_NAME (clear) (struct DYNARRAY_STRUCT *list)
{
  /* free__elements__ does nothing if the list is in the failed
     state.  */
  DYNARRAY_NAME (free__elements__)
    (list->u.dynarray_header.array, list->u.dynarray_header.used);
  list->u.dynarray_header.used = 0;
}

#ifdef DYNARRAY_FINAL_TYPE
/* Transfer the dynamic array to a permanent location at *RESULT.
   Returns true on success on false on allocation failure.  In either
   case, *LIST is re-initialized and can be reused.  A NULL pointer is
   stored in *RESULT if LIST refers to an empty list.  On success, the
   pointer in *RESULT is heap-allocated and must be deallocated using
   free.  */
__attribute_maybe_unused__ __attribute_warn_unused_result__ __nonnull ((1, 2))
static bool
DYNARRAY_NAME (finalize) (struct DYNARRAY_STRUCT *list,
                          DYNARRAY_FINAL_TYPE *result)
{
  struct dynarray_finalize_result res;
  if (__libc_dynarray_finalize (&list->u.dynarray_abstract,
                                DYNARRAY_SCRATCH (list),
                                sizeof (DYNARRAY_ELEMENT), &res))
    {
      /* On success, the result owns all the data.  */
      DYNARRAY_NAME (init) (list);
      *result = (DYNARRAY_FINAL_TYPE) { res.array, res.length };
      return true;
    }
  else
    {
      /* On error, we need to free all data.  */
      DYNARRAY_FREE (list);
      errno = ENOMEM;
      return false;
    }
}
#else /* !DYNARRAY_FINAL_TYPE */
/* Transfer the dynamic array to a heap-allocated array and return a
   pointer to it.  The pointer is NULL if memory allocation fails, or
   if the array is empty, so this function should be used only for
   arrays which are known not be empty (usually because they always
   have a sentinel at the end).  If LENGTHP is not NULL, the array
   length is written to *LENGTHP.  *LIST is re-initialized and can be
   reused.  */
__attribute_maybe_unused__ __attribute_warn_unused_result__ __nonnull ((1))
static DYNARRAY_ELEMENT *
DYNARRAY_NAME (finalize) (struct DYNARRAY_STRUCT *list, size_t *lengthp)
{
  struct dynarray_finalize_result res;
  if (__libc_dynarray_finalize (&list->u.dynarray_abstract,
                                DYNARRAY_SCRATCH (list),
                                sizeof (DYNARRAY_ELEMENT), &res))
    {
      /* On success, the result owns all the data.  */
      DYNARRAY_NAME (init) (list);
      if (lengthp != NULL)
        *lengthp = res.length;
      return res.array;
    }
  else
    {
      /* On error, we need to free all data.  */
      DYNARRAY_FREE (list);
      errno = ENOMEM;
      return NULL;
    }
}
#endif /* !DYNARRAY_FINAL_TYPE */

/* Undo macro definitions.  */

#undef DYNARRAY_CONCAT0
#undef DYNARRAY_CONCAT1
#undef DYNARRAY_NAME
#undef DYNARRAY_SCRATCH
#undef DYNARRAY_HAVE_SCRATCH

#undef DYNARRAY_STRUCT
#undef DYNARRAY_ELEMENT
#undef DYNARRAY_PREFIX
#undef DYNARRAY_ELEMENT_FREE
#undef DYNARRAY_ELEMENT_INIT
#undef DYNARRAY_INITIAL_SIZE
#undef DYNARRAY_FINAL_TYPE
