#include <src/__support/alloc/alloc.h>
#include <src/__support/alloc/arena.h>

namespace LIBC_NAMESPACE_DECL {

#define CONCAT(a, b) a##b
#define EXPAND_AND_CONCAT(a, b) CONCAT(a, b)

#define ALLOCATOR EXPAND_AND_CONCAT(LIBC_CONF_ALLOC_TYPE, _allocator)

BaseAllocator *allocator = reinterpret_cast<BaseAllocator *>(&ALLOCATOR);

} // namespace LIBC_NAMESPACE_DECL
