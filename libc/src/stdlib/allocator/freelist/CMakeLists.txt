# Only use freelist malloc for baremetal targets.
add_entrypoint_object(
  @libc.src.stdlib.freelist_malloc
  SRCS
    freelist_malloc.cpp
  HDRS
    malloc.h
  DEPENDS
    libc.src.__support.freelist_heap
)

get_target_property(freelist_malloc_is_skipped libc.src.stdlib.freelist_malloc "SKIPPED")

add_entrypoint_object(
    @libc.src.stdlib.malloc
    ALIAS
    DEPENDS
      .freelist_malloc
)

add_entrypoint_external(
  @libc.src.stdlib.free
)
add_entrypoint_external(
  @libc.src.stdlib.calloc
)
add_entrypoint_external(
  @libc.src.stdlib.realloc
)
add_entrypoint_external(
  @libc.src.stdlib.aligned_alloc
)
