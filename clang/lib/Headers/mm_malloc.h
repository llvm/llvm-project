/*===---- mm_malloc.h - Allocating and Freeing Aligned Memory Blocks -------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __MM_MALLOC_H
#define __MM_MALLOC_H

#include <stdlib.h>

#ifdef _WIN32
#include <malloc.h>
#else
#ifndef __cplusplus
extern int posix_memalign(void **memptr, size_t alignment, size_t size);
#else
// Some systems (e.g. those with GNU libc) declare posix_memalign with an
// exception specifier. Via an "egregious workaround" in
// Sema::CheckEquivalentExceptionSpec, Clang accepts the following as a valid
// redeclaration of glibc's declaration.
extern "C" int posix_memalign(void **memptr, size_t alignment, size_t size);
#endif
#endif

#if !(defined(_WIN32) && defined(_mm_malloc))
static __inline__ void *__attribute__((__always_inline__, __nodebug__,
                                       __malloc__))
_mm_malloc(size_t size, size_t align)
{
  if (align == 1) {
    return malloc(size);
  }

  if (!(align & (align - 1)) && align < sizeof(void *))
    align = sizeof(void *);

  void *mallocedMemory;
#ifdef _WIN32
  mallocedMemory = _aligned_malloc(size, align);
#else
  if (posix_memalign(&mallocedMemory, align, size))
    return 0;
#endif

  return mallocedMemory;
}

static __inline__ void __attribute__((__always_inline__, __nodebug__))
_mm_free(void *p)
{
  free(p);
}
#endif

#endif /* __MM_MALLOC_H */
