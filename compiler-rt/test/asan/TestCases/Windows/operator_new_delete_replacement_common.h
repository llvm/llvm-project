#include <new>
#include <stdio.h>

// If users provide fallback, all parents must be replaced as well.
// That is, you can't sanely provide a non-forwarding array op new without also replacing scalar op new.

// With that in mind, we need to cover the following scenarios for operator new:
// 1. array (asan)                    -> scalar (custom)
// 2. nothrow (asan)                  -> scalar (custom)
// 3. array nothrow (asan)            -> array  (custom)
// 4. array nothrow (asan)            -> array  (asan)           -> scalar (custom)
// 5. aligned array (asan)            -> aligned scalar (custom)
// 6. aligned nothrow (asan)          -> aligned scalar (custom)
// 7. aligned array nothrow (asan)    -> aligned array  (custom)
// 8. aligned array nothrow (asan)    -> aligned array  (asan)   -> aligned scalar (custom)

// And the following for operator delete:
// 9.  array (asan)                    -> scalar (custom)
// 10. nothrow (asan)                  -> scalar (custom)
// 11. sized (asan)                    -> scalar (custom) ** original bug report scenario **
// 12. sized array (asan)              -> array (custom)
// 13. sized array (asan)              -> array (asan)            -> scalar (custom)
// 14. array nothrow (asan)            -> array (custom)
// 15. array nothrow (asan)            -> array (asan)            -> scalar (custom)
// 16. aligned array (asan)            -> aligned scalar (custom)
// 17. aligned nothrow (asan)          -> aligned scalar (custom)
// 18. aligned sized (asan)            -> aligned scalar (custom)
// 19. aligned sized array (asan)      -> aligned array (custom)
// 20. aligned sized array (asan)      -> aligned array (asan)    -> aligned scalar (custom)
// 21. aligned array nothrow (asan)    -> aligned array (custom)
// 22. aligned array nothrow (asan)    -> aligned array (asan)    -> aligned scalar (custom)

#ifdef VERBOSE
#  define PRINTF(...) printf(__VA_ARGS__)
#else
#  define PRINTF(...)
#endif

template <size_t N> class arena {
public:
  void *alloc(const size_t size, const std::align_val_t al) {
    return alloc(size, static_cast<size_t>(al));
  }

  void *
  alloc(const size_t size,
        const size_t requested_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    if (requested_alignment == 0 ||
        (requested_alignment & (requested_alignment - 1))) {
      // Alignment must be non-zero and power of two.
      PRINTF("Allocation of size '%zu' alignment '%zu' failed due to bad "
             "arguments.\n",
             size, requested_alignment);
      throw std::bad_alloc{};
    }

    const size_t alignment =
        (requested_alignment <= __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            ? __STDCPP_DEFAULT_NEW_ALIGNMENT__
            : requested_alignment;

    // Adjust for alignment
    const size_t alignment_mask = alignment - 1;
    m_cur = reinterpret_cast<char *>(
        reinterpret_cast<unsigned __int64>(m_cur + alignment_mask) &
        ~alignment_mask);
    const size_t memory_block_size = (size + alignment_mask) & ~alignment_mask;

    if (m_cur + memory_block_size > m_buffer + N) {
      PRINTF("Allocation of size '%zu' alignment '%zu' failed due to out of "
             "memory.\n",
             size, requested_alignment);
      throw std::bad_alloc{};
    }

    char *const returned_memory_block = m_cur;
    m_cur += memory_block_size;

    PRINTF("Allocated '0x%p' of size '%zu' (requested '%zu') with alignment "
           "'%zu' (requested '%zu')\n",
           returned_memory_block, memory_block_size, size, alignment,
           requested_alignment);

    return returned_memory_block;
  }

  void free(const void *ptr) { PRINTF("Deallocated '0x%p'\n", ptr); }

private:
  char m_buffer[N];
  char *m_cur = m_buffer;
};

arena<100000> mem;

////////////////////////////////////
// clang-format off
// new() Fallback Ordering
//
// +----------+
// |new_scalar<---------------+
// +----^-----+               |
//      |                     |
// +----+-------------+  +----+----+
// |new_scalar_nothrow|  |new_array|
// +------------------+  +----^----+
//                            |
//               +------------+----+
//               |new_array_nothrow|
//               +-----------------+
// clang-format on

#if (DEFINED_REPLACEMENTS & SCALAR_NEW)
void *operator new(const size_t sz) {
  puts("new_scalar");
  return mem.alloc(sz);
}
#endif // MISSING_SCALAR_NEW

#if (DEFINED_REPLACEMENTS & SCALAR_NEW_NOTHROW)
void *operator new(const size_t sz, const std::nothrow_t &) noexcept {
  puts("new_scalar_nothrow");
  try {
    return mem.alloc(sz);
  } catch (...) {
    return nullptr;
  }
}
#endif // MISSING_SCALAR_NEW_NOTHROW

#if (DEFINED_REPLACEMENTS & ARRAY_NEW)
void *operator new[](const size_t sz) {
  puts("new_array");
  return mem.alloc(sz);
}
#endif // MISSING_ARRAY_NEW

#if (DEFINED_REPLACEMENTS & ARRAY_NEW_NOTHROW)
void *operator new[](const size_t sz, const std::nothrow_t &) noexcept {
  puts("new_array_nothrow");
  try {
    return mem.alloc(sz);
  } catch (...) {
    return nullptr;
  }
}
#endif // MISSING_ARRAY_NEW_NOTHROW

////////////////////////////////////////////////
// clang-format off
// Aligned new() Fallback Ordering
//
// +----------------+
// |new_scalar_align<--------------+
// +----^-----------+              |
//      |                          |
// +----+-------------------+  +---+-----------+
// |new_scalar_align_nothrow|  |new_array_align|
// +------------------------+  +---^-----------+
//                                 |
//                     +-----------+-----------+
//                     |new_array_align_nothrow|
//                     +-----------------------+
// clang-format on

#if (DEFINED_REPLACEMENTS & SCALAR_ALIGNED_NEW)
void *operator new(const size_t sz, const std::align_val_t al) {
  puts("new_scalar_align");
  return mem.alloc(sz, al);
}
#endif // MISSING_SCALAR_ALIGNED_NEW

#if (DEFINED_REPLACEMENTS & SCALAR_ALIGNED_NEW_NOTHROW)
void *operator new(const size_t sz, const std::align_val_t al,
                   const std::nothrow_t &) noexcept {
  puts("new_scalar_align_nothrow");
  try {
    return mem.alloc(sz, al);
  } catch (...) {
    return nullptr;
  }
}
#endif // MISSING_SCALAR_NEW_ALIGNED_NOTHROW

#if (DEFINED_REPLACEMENTS & ARRAY_ALIGNED_NEW)
void *operator new[](const size_t sz, const std::align_val_t al) {
  puts("new_array_align");
  return mem.alloc(sz, al);
}
#endif // MISSING_ARRAY_ALIGNED_NEW

#if (DEFINED_REPLACEMENTS & ARRAY_ALIGNED_NEW_NOTHROW)
void *operator new[](const size_t sz, const std::align_val_t al,
                     const std::nothrow_t &) noexcept {
  puts("new_array_align_nothrow");
  try {
    return mem.alloc(sz, al);
  } catch (...) {
    return nullptr;
  }
}
#endif // MISSING_ARRAY_ALIGNED_NEW_NOTHROW

////////////////////////////////////////////////////////////////
// clang-format off
// delete() Fallback Ordering
//
// +-------------+
// |delete_scalar<----+-----------------------+
// +--^----------+    |                       |
//    |               |                       |
// +--+---------+  +--+---------------+  +----+----------------+
// |delete_array|  |delete_scalar_size|  |delete_scalar_nothrow|
// +--^----^----+  +------------------+  +---------------------+
//    |    |
//    |    +-------------------+
//    |                        |
// +--+--------------+  +------+-------------+
// |delete_array_size|  |delete_array_nothrow|
// +-----------------+  +--------------------+
// clang-format on

#if (DEFINED_REPLACEMENTS & SCALAR_DELETE)
void operator delete(void *const ptr) noexcept {
  puts("delete_scalar");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_DELETE)
void operator delete[](void *const ptr) noexcept {
  puts("delete_array");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_SIZED_DELETE)
void operator delete[](void *const ptr, const size_t sz) noexcept {
  puts("delete_array_size");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_SIZED_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_DELETE_NOTHROW)
void operator delete[](void *const ptr, const std::nothrow_t &) noexcept {
  puts("delete_array_nothrow");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_DELETE_NOTHROW

#if (DEFINED_REPLACEMENTS & SCALAR_SIZED_DELETE)
void operator delete(void *const ptr, const size_t sz) noexcept {
  puts("delete_scalar_size");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_SIZED_DELETE

#if (DEFINED_REPLACEMENTS & SCALAR_DELETE_NOTHROW)
void operator delete(void *const ptr, const std::nothrow_t &) noexcept {
  puts("delete_scalar_nothrow");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_DELETE_NOTHROW

//////////////////////////////////////////////////////////////////////////////////
// clang-format off
// Aligned delete() Fallback Ordering
//
// +-------------------+
// |delete_scalar_align<----+---------------------------+
// +--^----------------+    |                           |
//    |                     |                           |
// +--+---------------+  +--+---------------------+  +--+------------------------+
// |delete_array_align|  |delete_scalar_size_align|  |delete_scalar_align_nothrow|
// +--^-----^---------+  +------------------------+  +---------------------------+
//    |     |
//    |     +------------------------+
//    |                              |
// +--+--------------------+  +------+-------------------+
// |delete_array_size_align|  |delete_array_align_nothrow|
// +-----------------------+  +--------------------------+
// clang-format on

#if (DEFINED_REPLACEMENTS & SCALAR_ALIGNED_DELETE)
void operator delete(void *const ptr, const std::align_val_t) noexcept {
  puts("delete_scalar_align");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_ALIGNED_DELETE)
void operator delete[](void *const ptr, const std::align_val_t) noexcept {
  puts("delete_array_align");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_SIZED_ALIGNED_DELETE)
void operator delete[](void *const ptr, const size_t sz,
                       const std::align_val_t) noexcept {
  puts("delete_array_size_align");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_SIZED_DELETE

#if (DEFINED_REPLACEMENTS & ARRAY_ALIGNED_DELETE_NOTHROW)
void operator delete[](void *const ptr, const std::align_val_t,
                       const std::nothrow_t &) noexcept {
  puts("delete_array_align_nothrow");
  mem.free(ptr);
}
#endif // MISSING_ARRAY_DELETE_NOTHROW

#if (DEFINED_REPLACEMENTS & SCALAR_SIZED_ALIGNED_DELETE)
void operator delete(void *const ptr, const size_t sz,
                     const std::align_val_t) noexcept {
  puts("delete_scalar_size_align");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_SIZED_DELETE

#if (DEFINED_REPLACEMENTS & SCALAR_ALIGNED_DELETE_NOTHROW)
void operator delete(void *const ptr, const std::align_val_t,
                     const std::nothrow_t &) noexcept {
  puts("delete_scalar_align_nothrow");
  mem.free(ptr);
}
#endif // MISSING_SCALAR_DELETE_NOTHROW

// Explicitly call delete so we can explicitly choose sized vs non-sized versions of each.
// Also provide explicit nothrow version, since that can't be implicitly invoked.
template <typename T> void op_delete_scalar(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete(ptr, std::align_val_t{alignof(T)});
  } else {
    operator delete(ptr);
  }
}

template <typename T> void op_delete_array(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete[](ptr, std::align_val_t{alignof(T)});
  } else {
    operator delete[](ptr);
  }
}

template <typename T> void op_delete_scalar_nothrow(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete(ptr, std::align_val_t{alignof(T)}, std::nothrow_t{});
  } else {
    operator delete(ptr, std::nothrow_t{});
  }
}

template <typename T> void op_delete_array_nothrow(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete[](ptr, std::align_val_t{alignof(T)}, std::nothrow_t{});
  } else {
    operator delete[](ptr, std::nothrow_t{});
  }
}

template <typename T> void op_delete_scalar_size(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete(ptr, sizeof(T), std::align_val_t{alignof(T)});
  } else {
    operator delete(ptr, sizeof(T));
  }
}

template <size_t N, typename T> void op_delete_array_size(T *ptr) {
  if (alignof(T) > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
    operator delete[](ptr, sizeof(T) * N, std::align_val_t{alignof(T)});
  } else {
    operator delete[](ptr, sizeof(T) * N);
  }
}

template <typename T> void test_allocations() {
  T *scalar = new T();
  T *array = new T[5];

  T *scalar_nothrow = new (std::nothrow) T();
  T *array_nothrow = new (std::nothrow) T[5];

  op_delete_scalar(scalar);
  op_delete_array(array);

  op_delete_scalar_nothrow(scalar_nothrow);
  op_delete_array_nothrow(array_nothrow);

  T *scalar_size = new T();
  T *array_size = new T[5];

  op_delete_scalar_size(scalar_size);
  op_delete_array_size<5>(array_size);
}

struct alignas(32) overaligned {
  double a;
};

#ifdef TEST_DLL
extern "C" __declspec(dllexport) int test_function()
#else
int main()
#endif
{
  test_allocations<int>();
  test_allocations<overaligned>();
  return 0;
}
