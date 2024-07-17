#ifndef LIBC_FREELIST_MALLOC_SIZE
#error "LIBC_FREELIST_MALLOC_SIZE was not defined for this build."
#endif
.weak __libc_heap_limit
.set __libc_heap_limit, LIBC_FREELIST_MALLOC_SIZE
