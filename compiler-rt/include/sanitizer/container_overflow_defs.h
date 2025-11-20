//===-- sanitizer/container_overflow_defs.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Public sanitizer interface defs for container overflow checks.
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_CONTAINER_OVERFLOW_DEFS_H
#define SANITIZER_CONTAINER_OVERFLOW_DEFS_H

// Windows allows a user to set their default calling convention, but we always
// use __cdecl
#ifdef _WIN32
#define SANITIZER_CDECL __cdecl
#else
#define SANITIZER_CDECL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Annotates the current state of a contiguous container, such as
/// <c>std::vector</c>, <c>std::string</c>, or similar.
///
/// A contiguous container is a container that keeps all of its elements
/// in a contiguous region of memory. The container owns the region of memory
/// <c>[beg, end)</c>; the memory <c>[beg, mid)</c> is used to store the
/// current elements, and the memory <c>[mid, end)</c> is reserved for future
/// elements (<c>beg <= mid <= end</c>). For example, in
/// <c>std::vector<> v</c>:
///
/// \code
///   beg = &v[0];
///   end = beg + v.capacity() * sizeof(v[0]);
///   mid = beg + v.size()     * sizeof(v[0]);
/// \endcode
///
/// This annotation tells the Sanitizer tool about the current state of the
/// container so that the tool can report errors when memory from
/// <c>[mid, end)</c> is accessed. Insert this annotation into methods like
/// <c>push_back()</c> or <c>pop_back()</c>. Supply the old and new values of
/// <c>mid</c>(<c><i>old_mid</i></c> and <c><i>new_mid</i></c>). In the initial
/// state <c>mid == end</c>, so that should be the final state when the
/// container is destroyed or when the container reallocates the storage.
///
/// For ASan, <c><i>beg</i></c> no longer needs to be 8-aligned,
/// first and last granule may be shared with other objects
/// and therefore the function can be used for any allocator.
///
/// The following example shows how to use the function:
///
/// \code
///   int32_t x[3]; // 12 bytes
///   char *beg = (char*)&x[0];
///   char *end = beg + 12;
///   __sanitizer_annotate_contiguous_container(beg, end, beg, end);
/// \endcode
///
/// \note  Use this function with caution and do not use for anything other
/// than vector-like classes.
/// \note  Unaligned <c><i>beg</i></c> or <c><i>end</i></c> may miss bugs in
/// these granules.
///
/// \param __beg Beginning of memory region.
/// \param __end End of memory region.
/// \param __old_mid Old middle of memory region.
/// \param __new_mid New middle of memory region.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline void SANITIZER_CDECL
__sanitizer_annotate_contiguous_container(const void *__beg, const void *__end,
                                          const void *__old_mid,
                                          const void *__new_mid) {}
#else
void SANITIZER_CDECL __sanitizer_annotate_contiguous_container(
    const void *__beg, const void *__end, const void *__old_mid,
    const void *__new_mid);
#endif

/// Similar to <c>__sanitizer_annotate_contiguous_container</c>.
///
/// Annotates the current state of a contiguous container memory,
/// such as <c>std::deque</c>'s single chunk, when the boundries are moved.
///
/// A contiguous chunk is a chunk that keeps all of its elements
/// in a contiguous region of memory. The container owns the region of memory
/// <c>[storage_beg, storage_end)</c>; the memory <c>[container_beg,
/// container_end)</c> is used to store the current elements, and the memory
/// <c>[storage_beg, container_beg), [container_end, storage_end)</c> is
/// reserved for future elements (<c>storage_beg <= container_beg <=
/// container_end <= storage_end</c>). For example, in <c> std::deque </c>:
/// - chunk with a frist deques element will have container_beg equal to address
///  of the first element.
/// - in every next chunk with elements, true is  <c> container_beg ==
/// storage_beg </c>.
///
/// Argument requirements:
/// During unpoisoning memory of empty container (before first element is
/// added):
/// - old_container_beg_p == old_container_end_p
/// During poisoning after last element was removed:
/// - new_container_beg_p == new_container_end_p
/// \param __storage_beg Beginning of memory region.
/// \param __storage_end End of memory region.
/// \param __old_container_beg Old beginning of used region.
/// \param __old_container_end End of used region.
/// \param __new_container_beg New beginning of used region.
/// \param __new_container_end New end of used region.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline void
    SANITIZER_CDECL __sanitizer_annotate_double_ended_contiguous_container(
        const void *__storage_beg, const void *__storage_end,
        const void *__old_container_beg, const void *__old_container_end,
        const void *__new_container_beg, const void *__new_container_end) {}
#else
void SANITIZER_CDECL __sanitizer_annotate_double_ended_contiguous_container(
    const void *__storage_beg, const void *__storage_end,
    const void *__old_container_beg, const void *__old_container_end,
    const void *__new_container_beg, const void *__new_container_end);
#endif

/// Copies memory annotations from a source storage region to a destination
/// storage region. After the operation, the destination region has the same
/// memory annotations as the source region, as long as sanitizer limitations
/// allow it (more bytes may be unpoisoned than in the source region, resulting
/// in more false negatives, but never false positives). If the source and
/// destination regions overlap, only the minimal required changes are made to
/// preserve the correct annotations. Old storage bytes that are not in the new
/// storage should have the same annotations, as long as sanitizer limitations
/// allow it.
///
/// This function is primarily designed to be used when moving trivially
/// relocatable objects that may have poisoned memory, making direct copying
/// problematic under sanitizer. However, this function does not move memory
/// content itself, only annotations.
///
/// A contiguous container is a container that keeps all of its elements in a
/// contiguous region of memory. The container owns the region of memory
/// <c>[src_begin, src_end)</c> and <c>[dst_begin, dst_end)</c>. The memory
/// within these regions may be alternately poisoned and non-poisoned, with
/// possibly smaller poisoned and unpoisoned regions.
///
/// If this function fully poisons a granule, it is marked as "container
/// overflow".
///
/// Argument requirements: The destination container must have the same size as
/// the source container, which is inferred from the beginning and end of the
/// source region. Addresses may be granule-unaligned, but this may affect
/// performance.
///
/// \param __src_begin Begin of the source container region.
/// \param __src_end End of the source container region.
/// \param __dst_begin Begin of the destination container region.
/// \param __dst_end End of the destination container region.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline void SANITIZER_CDECL
__sanitizer_copy_contiguous_container_annotations(const void *__src_begin,
                                                  const void *__src_end,
                                                  const void *__dst_begin,
                                                  const void *__dst_end) {}
#else
void SANITIZER_CDECL __sanitizer_copy_contiguous_container_annotations(
    const void *__src_begin, const void *__src_end, const void *__dst_begin,
    const void *__dst_end);
#endif

/// Returns true if the contiguous container <c>[beg, end)</c> is properly
/// poisoned.
///
/// Proper poisoning could occur, for example, with
/// <c>__sanitizer_annotate_contiguous_container</c>), that is, if
/// <c>[beg, mid)</c> is addressable and <c>[mid, end)</c> is unaddressable.
/// Full verification requires O (<c>end - beg</c>) time; this function tries
/// to avoid such complexity by touching only parts of the container around
/// <c><i>beg</i></c>, <c><i>mid</i></c>, and <c><i>end</i></c>.
///
/// \param __beg Beginning of memory region.
/// \param __mid Middle of memory region.
/// \param __end Old end of memory region.
///
/// \returns True if the contiguous container <c>[beg, end)</c> is properly
///  poisoned.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline int SANITIZER_CDECL
__sanitizer_verify_contiguous_container(const void *__beg, const void *__mid,
                                        const void *__end) {}
#else
int SANITIZER_CDECL __sanitizer_verify_contiguous_container(const void *__beg,
                                                            const void *__mid,
                                                            const void *__end);
#endif

/// Returns true if the double ended contiguous
/// container <c>[storage_beg, storage_end)</c> is properly poisoned.
///
/// Proper poisoning could occur, for example, with
/// <c>__sanitizer_annotate_double_ended_contiguous_container</c>), that is, if
/// <c>[storage_beg, container_beg)</c> is not addressable, <c>[container_beg,
/// container_end)</c> is addressable and <c>[container_end, end)</c> is
/// unaddressable. Full verification requires O (<c>storage_end -
/// storage_beg</c>) time; this function tries to avoid such complexity by
/// touching only parts of the container around <c><i>storage_beg</i></c>,
/// <c><i>container_beg</i></c>, <c><i>container_end</i></c>, and
/// <c><i>storage_end</i></c>.
///
/// \param __storage_beg Beginning of memory region.
/// \param __container_beg Beginning of used region.
/// \param __container_end End of used region.
/// \param __storage_end End of memory region.
///
/// \returns True if the double-ended contiguous container <c>[storage_beg,
/// container_beg, container_end, end)</c> is properly poisoned - only
/// [container_beg; container_end) is addressable.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline int
    SANITIZER_CDECL __sanitizer_verify_double_ended_contiguous_container(
        const void *__storage_beg, const void *__container_beg,
        const void *__container_end, const void *__storage_end) {}
#else
int SANITIZER_CDECL __sanitizer_verify_double_ended_contiguous_container(
    const void *__storage_beg, const void *__container_beg,
    const void *__container_end, const void *__storage_end);
#endif

/// Similar to <c>__sanitizer_verify_contiguous_container()</c> but also
/// returns the address of the first improperly poisoned byte.
///
/// Returns NULL if the area is poisoned properly.
///
/// \param __beg Beginning of memory region.
/// \param __mid Middle of memory region.
/// \param __end Old end of memory region.
///
/// \returns The bad address or NULL.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline const void *SANITIZER_CDECL
__sanitizer_contiguous_container_find_bad_address(const void *__beg,
                                                  const void *__mid,
                                                  const void *__end) {}
#else
const void *SANITIZER_CDECL __sanitizer_contiguous_container_find_bad_address(
    const void *__beg, const void *__mid, const void *__end);
#endif

/// returns the address of the first improperly poisoned byte.
///
/// Returns NULL if the area is poisoned properly.
///
/// \param __storage_beg Beginning of memory region.
/// \param __container_beg Beginning of used region.
/// \param __container_end End of used region.
/// \param __storage_end End of memory region.
///
/// \returns The bad address or NULL.
#ifdef __SANITIZER_DISABLE_CONTAINER_OVERFLOW__
__attribute__((__internal_linkage__)) inline const void *SANITIZER_CDECL
__sanitizer_double_ended_contiguous_container_find_bad_address(
    const void *__storage_beg, const void *__container_beg,
    const void *__container_end, const void *__storage_end) {}
#else
const void *SANITIZER_CDECL
__sanitizer_double_ended_contiguous_container_find_bad_address(
    const void *__storage_beg, const void *__container_beg,
    const void *__container_end, const void *__storage_end);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_CONTAINER_OVERFLOW_DEFS_H
