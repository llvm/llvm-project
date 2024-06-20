#include "include/llvm-libc-macros/stdint-macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/utility/move.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/string/memory_utils/inline_memmove.h"

namespace LIBC_NAMESPACE {

template <typename T, unsigned Factor> class TreeNode {

  struct PositionInfo {
    // Usage of the key slots.
    unsigned short usage;
    // The index of current node in the parent node.
    unsigned short parent_index;
  };

  // The layout of the key_nodes array is as follows:
  // ========================================================================
  // | parent  | key 0   | key 1 |  .......  |  key 2*B    | position info  |
  // |---------|---------|-------|-----------|-------------|----------------|
  // | child 0 | child 1 | ..... | child 2*B | child 2*B+1 | extra sentinel |
  // ========================================================================
  // - The parent pointer is stored in the first key node.
  // - The position info is stored in the last key node.
  // - For each key, its left child is stored in the previous key node and its
  //   right child is stored in the same key node.
  // - After the last valid key, the child field is marked as -1 to indicate
  //   the end of the keys. When all available keys are used, such -1 will
  //   fall into the extra sentinel field.
  //
  // A B-Tree node represents a flattened binary tree.
  //
  //     key 0
  //   ./      \.
  // child 0   key 1
  //         ./     \.
  //     child 1    key 2
  //              ./     \.
  //           ....       ....
  struct KeyNode {
    union {
      TreeNode *parent = nullptr;
      T key;
      PositionInfo info;
    };
    union {
      TreeNode *child = nullptr;
      intptr_t sentinel;
    };
  } key_nodes[2 * Factor + 1];

  LIBC_INLINE TreeNode *&parent() const { return key_nodes[0].parent; }
  LIBC_INLINE TreeNode *&child(unsigned index) {
    return key_nodes[index].child;
  }
  LIBC_INLINE unsigned short &usage() {
    return key_nodes[2 * Factor].info.usage;
  }
  LIBC_INLINE unsigned short &parent_index() {
    return key_nodes[2 * Factor].info.parent_index;
  }
  LIBC_INLINE T &key(unsigned index) { return key_nodes[index + 1].key; }
  LIBC_INLINE bool is_internal() const { return key_nodes[0].child != nullptr; }

  // Move elements towards the end of the array. The index is with respect to
  // the key slot.
  LIBC_INLINE void move_backward(unsigned short index) {
    LIBC_ASSERT(usage() < 2 * Factor);
    key_nodes[usage() + 2].sentinel = -1;
    if constexpr (__is_trivially_copyable(T))
      inline_memmove(&key_nodes[index + 2], &key_nodes[index + 1],
                     (usage() - index) * sizeof(KeyNode));
    else
      for (unsigned i = usage(); i > index; --i) {
        new (&key_nodes[i + 1].key) T(cpp::move(key(i)));
        key_nodes[i + 1].child = key_nodes[i].child;
        key_nodes[i].key.~T();
      }
  }

  class SearchResult {
    bool found_flag;
    unsigned data;
    LIBC_INLINE SearchResult(unsigned data, bool found_flag)
        : found_flag(found_flag), data(data) {}

  public:
    LIBC_INLINE static SearchResult found(unsigned index) {
      return SearchResult(index, true);
    }
    LIBC_INLINE static SearchResult go_down(unsigned index) {
      return SearchResult(index, false);
    }
    LIBC_INLINE bool is_found() const { return found_flag; }
    LIBC_INLINE operator unsigned() const { return data; }
  };

  template <typename Compare>
  LIBC_INLINE SearchResult local_search(const T &target,
                                        const Compare &cmp) const {
    unsigned idx = 0;
    int last_cmp = 0;
    while (idx < usage()) {
      last_cmp = cmp(key(idx), target);
      if (last_cmp >= 0)
        break;
      ++idx;
    }
    if (idx == usage() || last_cmp > 0)
      return SearchResult::go_down(idx);
    return SearchResult::found(idx);
  }

  template <typename Compare>
  T *find(const T &target, const Compare &cmp) const {
    LIBC_ASSERT(usage() < 2 * Factor);
    TreeNode *node = this;
    while (node) {
      SearchResult result = node->local_search(target, cmp);
      if (result.is_found())
        return &node->key(result);
      node = node->child(result);
    }
    return nullptr;
  }

  struct SplitResult {
    T median;
    TreeNode *left, *right;
  };

  // This allocation buffer is used hold dynamically allocated tree nodes.
  // The btree insertion path needs to support fallible allocation. Therefore,
  // new nodes are allocated in batches. In the worst case, the number of
  // allocations should be proportional to the height of the tree, which should
  // not be too large. Hence, the buffer will be reserved on the stack.
  class AllocationBuffer {
    TreeNode **buffer;

    LIBC_INLINE AllocationBuffer(TreeNode **buffer) : buffer(buffer) {}

  public:
    LIBC_INLINE TreeNode *next() { return *buffer++; }

    template <class F>
    LIBC_INLINE static bool with_allocations(unsigned size, F &&func) {
      TreeNode **buffer =
          static_cast<TreeNode **>(__builtin_alloca(size * sizeof(TreeNode *)));
      unsigned allocated;
      // allocate nodes in batch
      for (allocated = 0; allocated < size; ++allocated) {
        AllocChecker ac;
        buffer[allocated] = new (ac) TreeNode();
        if (!ac)
          break;
      }
      // On failure, squash all existing allocations.
      if (allocated < size) {
        for (unsigned i = 0; i < allocated; ++i)
          delete buffer[i];
        return false;
      }
      AllocationBuffer alloc_buffer(buffer);
      func(alloc_buffer);
      return true;
    }
  };

  LIBC_INLINE TreeNode() : key_nodes{} {}
};

} // namespace LIBC_NAMESPACE
