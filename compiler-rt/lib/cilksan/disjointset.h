/* -*- Mode: C++ -*- */

#ifndef _DISJOINTSET_H
#define _DISJOINTSET_H

#include <cstdio>
#include <cstdlib>
#include <stdint.h>

#include "aligned_alloc.h"
#include "list.h"
#include "debug_util.h"
#include "spbag.h"

extern List_t disjoint_set_list;

#if CILKSAN_DEBUG
static int64_t DS_ID = 0;
#endif

template <typename DISJOINTSET_DATA_T>
class DisjointSet_t {
private:
  // the node that initialized this set; const field that does not change
  DISJOINTSET_DATA_T const _node;
  // the oldest node representing the set that the _node belongs to
  DISJOINTSET_DATA_T _set_node;
  DisjointSet_t *_set_parent = nullptr;
  uint64_t _rank; // roughly as the height of this node

  int64_t _ref_count;

#if CILKSAN_DEBUG
  // HACK: The destructor calls a callback to free the set node, but in order
  // for that callback to get the set node, it needs to call find_set which has
  // assertions for ref counts. Thus, we don't dec our ref count if we're
  // destructing.
  bool _destructing;
#endif

  __attribute__((always_inline))
  void assert_not_freed() const {
    WHEN_CILKSAN_DEBUG(cilksan_assert(_destructing || _ref_count >= 0));
  }

  /*
   * The only reason we need this function is to ensure that the _set_node
   * returned for representing this set is the oldest node in the set.
   */
  __attribute__((always_inline))
  void swap_set_node_with(DisjointSet_t *that) {
    assert_not_freed();
    that->assert_not_freed();
    DISJOINTSET_DATA_T tmp;
    tmp = this->_set_node;
    this->_set_node = that->_set_node;
    that->_set_node = tmp;
  }

  // Frees the old parent if it has no more references.
  __attribute__((always_inline))
  void set_parent(DisjointSet_t *that) {
    assert_not_freed();

    DisjointSet_t *old_parent = this->_set_parent;

    this->_set_parent = that;
    that->inc_ref_count();

    // dec_ref_count checks whether a node is its only reference (through
    // parent). If we called dec_ref_count (removing the parent relationship)
    // before setting this's parent and we had another reference besides the
    // parent relationship, dec_ref_count would incorrectly believe that this's
    // only reference is in having itself as a parent.
    cilksan_assert(old_parent != NULL);

    old_parent->dec_ref_count();
    DBG_TRACE(DEBUG_DISJOINTSET, "DS %ld points to DS %ld\n", _ID, that->_ID);
    DBG_TRACE(DEBUG_DISJOINTSET, "DS %ld refcnt %ld\n",
              that->_ID, that->_ref_count);
  }

  /*
   * Links this disjoint set to that disjoint set.
   * Don't need to be public.
   *
   * @param that that disjoint set.
   */
  __attribute__((always_inline))
  void link(DisjointSet_t *that) {
    assert_not_freed();
    cilksan_assert(that != NULL);

    // link the node with smaller height into the node with larger height
    if (this->_rank > that->_rank) {
      that->set_parent(this);
    } else {
      this->set_parent(that);
      if (this->_rank == that->_rank)
	++that->_rank;
      // because we are linking this into the forest rooted at that, let's
      // swap the nodes in this object and that object to keep the metadata
      // hold in the node consistent.
      this->swap_set_node_with(that);
    }
  }

  /*
   * Finds the set containing this disjoint set element.
   *
   * Note: Performs path compression along the way.
   *       The _set_parent field will be updated after the call.
   */
  __attribute__((always_inline))
  DisjointSet_t* find_set() {
    assert_not_freed();
    WHEN_CILKSAN_DEBUG(cilksan_assert(!_destructing));

    DisjointSet_t *node = this;
    node->assert_not_freed();
    DisjointSet_t *parent = node->_set_parent;
    cilksan_assert(parent);
    if (parent->_set_parent == parent)
      return parent;
    // // Fast test to see if node is the set.
    // if (node->_set_parent == node)
    //   return node;

    // // Fast test to see if node->_set_parent is the set.
    // DisjointSet_t *parent = node->_set_parent;
    // cilksan_assert(parent);
    // if (parent->_set_parent == parent)
    //   return parent;

    // Both fast tests failed.  Traverse the list to get to the set, and do path
    // compression along the way.

    disjoint_set_list.lock();

#if CILKSAN_DEBUG
    int64_t tmp_ref_count = _ref_count;
#endif

    while (node->_set_parent != node) {
      cilksan_assert(node->_set_parent);

      // if (__builtin_expect(!_destructing || node != this, 1)) {
      //   disjoint_set_list.push(node);
      // }
      // disjoint_set_list.push(node);
      DisjointSet_t *prev = node;
      node = node->_set_parent;
      if (node->_set_parent != node)
        disjoint_set_list.push(prev);
    }

    cilksan_assert(tmp_ref_count == _ref_count);

    // node is now the root. Perform path compression by updating the parents
    // of each of the nodes we saw.
    // We process backwards so that in case a node ought to be freed (i.e. its
    // child was the last referencing it), we don't process it after freeing.
    for (int i = disjoint_set_list.length() - 1; i >= 0; i--) {
      DisjointSet_t *p = (DisjointSet_t *)disjoint_set_list.list()[i];
      // We don't need to check that p != p->_set_parent because the root of
      // the set wasn't pushed to the list (see the while loop above).
      p->set_parent(node);
    }

    disjoint_set_list.unlock();
    return node;
  }

  DisjointSet_t() = delete;
  DisjointSet_t(const DisjointSet_t &) = delete;
  DisjointSet_t(DisjointSet_t &&) = delete;

public:
#if CILKSAN_DEBUG
  int64_t _ID;
#endif

  explicit DisjointSet_t(DISJOINTSET_DATA_T node) :
      _node(node), _set_node(node), _set_parent(NULL), _rank(0), _ref_count(0)
#if CILKSAN_DEBUG
      , _destructing(false), _ID(DS_ID++)
#endif
  {
    this->_set_parent = this;
    this->inc_ref_count();

    DBG_TRACE(DEBUG_DISJOINTSET, "\nCreating DS %ld\n", _ID);
    WHEN_CILKSAN_DEBUG(debug_count++);
  }

#if CILKSAN_DEBUG
  static long debug_count;
  static uint64_t nodes_created;
#endif

  static void (*dtor_callback)(DisjointSet_t *);

  ~DisjointSet_t() {
    WHEN_CILKSAN_DEBUG(_destructing = true);
    dtor_callback(this);
    if (this->_set_parent != this) {
      // Otherwise, we run the risk of double freeing.
      _set_parent->dec_ref_count();
    }
    DBG_TRACE(DEBUG_DISJOINTSET, "Deleting DS %ld\n", _ID);

    WHEN_CILKSAN_DEBUG({
        _destructing = false;
        _set_parent = NULL;
        _ref_count = -1;

        debug_count--;
      });
  }

  // Decrements the ref count.  Returns true if the node was deleted
  // as a result.
  __attribute__((always_inline))
  int64_t dec_ref_count(int64_t count = 1) {
    assert_not_freed();
    cilksan_assert(_ref_count >= count);
    _ref_count -= count;
    DBG_TRACE(DEBUG_DISJOINTSET, "DS %ld refcnt %ld\n", _ID, _ref_count);
    if (_ref_count == 0 || (_ref_count == 1 && this->_set_parent == this)) {
      delete this;
      return 0;
    }
    return _ref_count;
  }

  __attribute__((always_inline))
  void inc_ref_count(int64_t count = 1) {
    assert_not_freed();

    _ref_count += count;
  }

  __attribute__((always_inline))
  DISJOINTSET_DATA_T get_node() const {
    assert_not_freed();

    return _node;
  }

  __attribute__((always_inline))
  DISJOINTSET_DATA_T get_set_node() {
    assert_not_freed();

    return find_set()->_set_node;
  }

  /*
   * Unions this disjoint set and that disjoint set.
   *
   * NOTE: Implicitly, in order to maintain the oldest _set_node, one should
   * always combine younger set into this set (defined by creation time).  Since
   * we union by rank, we may end up linking this set to the younger set.  To
   * make sure that we always return the oldest _node to represent the set, we
   * use an additional _set_node field to keep track of the oldest node and use
   * that to represent the set.
   *
   * @param that that (younger) disjoint set.
   */
  // Called "combine," because "union" is a reserved keyword in C
  __attribute__((always_inline))
  void combine(DisjointSet_t *that) {
    assert_not_freed();

    cilksan_assert(that);
    cilksan_assert(this->find_set() != that->find_set());
    this->find_set()->link(that->find_set());
    cilksan_assert(this->find_set() == that->find_set());
  }

  // Custom memory allocation for disjoint sets.
  struct DJSlab_t {
    // System-page size.
    static constexpr unsigned SYS_PAGE_SIZE = 4096;
    // Mask to get sub-system-page portion of a memory address.
    static constexpr uintptr_t SYS_PAGE_DATA_MASK = SYS_PAGE_SIZE - 1;
    // Mask to get the system page of a memory address.
    static constexpr uintptr_t SYS_PAGE_MASK = ~SYS_PAGE_DATA_MASK;

    DJSlab_t *Next = nullptr;
    DJSlab_t *Prev = nullptr;

    static constexpr int UsedMapSize = 2;
    uint64_t UsedMap[UsedMapSize] = { 0 };

    static const size_t NumDJSets =
      (SYS_PAGE_SIZE - (2 * sizeof(DJSlab_t *)) - sizeof(uint64_t[UsedMapSize]))
      / sizeof(DisjointSet_t);
    // DisjointSet_t DJSets[NumDJSets];
    char DJSets[NumDJSets * sizeof(DisjointSet_t)];

    DJSlab_t() {
      UsedMap[UsedMapSize-1] |= ~((1UL << (NumDJSets % 64)) - 1);
    }

    // Returns true if this slab contains no free lines.
    bool isFull() const {
      for (int i = 0; i < UsedMapSize; ++i)
        if (UsedMap[i] != static_cast<uint64_t>(-1))
          return false;
      return true;
    }

    // Get a free disjoint set from the slab, marking that disjoint set as used
    // in the process.  Returns nullptr if no free disjoint set is available.
    DisjointSet_t *getFreeDJSet() {
      for (int i = 0; i < UsedMapSize; ++i) {
        if (UsedMap[i] == static_cast<uint64_t>(-1))
          continue;

        // Get the free line.
        DisjointSet_t *DJSet = reinterpret_cast<DisjointSet_t *>(
            &DJSets[(64 * i + __builtin_ctzl(UsedMap[i] + 1)) * sizeof(DisjointSet_t)]);

        // Mark the line as used.
        UsedMap[i] |= UsedMap[i] + 1;

        return DJSet;
      }
      // No free lines in this slab.
      return nullptr;
    }

    // Returns a line to this slab, marking that line as available.
    void returnDJSet(DisjointSet_t *DJSet) {
      uintptr_t DJSetPtr = reinterpret_cast<uintptr_t>(DJSet);
      cilksan_assert(
          (DJSetPtr & SYS_PAGE_MASK) == reinterpret_cast<uintptr_t>(this) &&
          "Disjoint set does not belong to this slab.");

      // Compute the index of this line in the array.
      uint64_t DJSetIdx = DJSetPtr & SYS_PAGE_DATA_MASK;
      // DJSetIdx -= offsetof(DJSlab_t, DJSets);
      // FIXME: The following code assumes no padding between fields.
      DJSetIdx -= ((2 * sizeof(DJSlab_t *)) + sizeof(uint64_t[UsedMapSize]));
      DJSetIdx /= sizeof(DisjointSet_t);

      // Mark the line as available in the map.
      uint64_t MapIdx = DJSetIdx / 64;
      uint64_t MapBit = DJSetIdx % 64;

      cilksan_assert(MapIdx < UsedMapSize && "Invalid MapIdx.");
      cilksan_assert(0 != (UsedMap[MapIdx] & (1UL << MapBit)) &&
                     "Disjoint set is not marked used.");
      UsedMap[MapIdx] &= ~(1UL << MapBit);
    }
  };

  class DJSAllocator {
    DJSlab_t *FreeSlabs = nullptr;
    DJSlab_t *FullSlabs = nullptr;
  public:
    DJSAllocator() {
      FreeSlabs =
        new (my_aligned_alloc(DJSlab_t::SYS_PAGE_SIZE,
                              sizeof(DJSlab_t))) DJSlab_t;
    }

    ~DJSAllocator() {
      cilksan_assert(!FullSlabs && "Full slabs remaining.");
      // Destruct the free slabs and free their memory.
      DJSlab_t *Slab = FreeSlabs;
      DJSlab_t *PrevSlab = nullptr;
      while (Slab) {
        PrevSlab = Slab;
        Slab = Slab->Next;
        PrevSlab->~DJSlab_t();
        free(PrevSlab);
      }
      FreeSlabs = nullptr;
    }

    DisjointSet_t *getDJSet() {
      DJSlab_t *Slab = FreeSlabs;
      DisjointSet_t *DJSet = Slab->getFreeDJSet();

      // If Slab is now full, move it to the Full list.
      if (Slab->isFull()) {
        if (!Slab->Next)
          // Allocate a new slab if necessary.
          FreeSlabs = new (my_aligned_alloc(DJSlab_t::SYS_PAGE_SIZE,
                                            sizeof(DJSlab_t))) DJSlab_t;
        else {
          Slab->Next->Prev = nullptr;
          FreeSlabs = Slab->Next;
        }
        // Push slab to the beginning of the full list.
        Slab->Next = FullSlabs;
        if (FullSlabs)
          FullSlabs->Prev = Slab;
        FullSlabs = Slab;
      }

      cilksan_assert(DJSet && "No disjoinst set found.");
      return DJSet;
    }

    void freeDJSet(void *Ptr) {
      // Derive the pointer to the slab.
      DJSlab_t *Slab = reinterpret_cast<DJSlab_t *>(
          reinterpret_cast<uintptr_t>(Ptr) & DJSlab_t::SYS_PAGE_MASK);

      if (Slab->isFull()) {
        // Slab is no longer full, so move it back to the free list.
        if (Slab->Prev)
          Slab->Prev->Next = Slab->Next;
        else
          FullSlabs = Slab->Next;

        // Make Slab's successor point to Slab's predecessor.
        if (Slab->Next)
          Slab->Next->Prev = Slab->Prev;

        // Push Slab to the start of the free list.
        Slab->Prev = nullptr;
        Slab->Next = FreeSlabs;
        FreeSlabs->Prev = Slab;
        FreeSlabs = Slab;
      } else if (FreeSlabs != Slab) {
        // Remove Slab from its place in FreeSlabs.
        Slab->Prev->Next = Slab->Next;
        if (Slab->Next)
          Slab->Next->Prev = Slab->Prev;

        // Move Slab to the start of FreeSlabs.
        Slab->Prev = nullptr;
        Slab->Next = FreeSlabs;
        FreeSlabs->Prev = Slab;
        FreeSlabs = Slab;
      }

      Slab->returnDJSet(static_cast<DisjointSet_t *>(Ptr));
    }
  };

  // // Simple free-list allocator to conserve space and time in managing
  // // DisjointSet_t objects.
  // static DisjointSet_t *free_list;
  static DJSAllocator &Alloc;

  void *operator new(size_t size) {
    return Alloc.getDJSet();
    // if (free_list) {
    //   DisjointSet_t *new_node = free_list;
    //   free_list = free_list->_set_parent;
    //   return new_node;
    // }
    // return ::operator new(size);
  }

  void operator delete(void *ptr) {
    Alloc.freeDJSet(ptr);
    // DisjointSet_t *del_node = reinterpret_cast<DisjointSet_t *>(ptr);
    // del_node->_set_parent = free_list;
    // free_list = del_node;
  }

  // static void cleanup_freelist() {
  //   DisjointSet_t *node = free_list;
  //   DisjointSet_t *next = nullptr;
  //   while (node) {
  //     next = node->_set_parent;
  //     ::operator delete(node);
  //     node = next;
  //   }
  // }
};

// Explicit instantiations for cilksan.
template<>
void (*DisjointSet_t<SPBagInterface *>::dtor_callback)(DisjointSet_t *);

// template<>
// DisjointSet_t<SPBagInterface *> *DisjointSet_t<SPBagInterface *>::free_list;
template<>
DisjointSet_t<SPBagInterface *>::DJSAllocator
&DisjointSet_t<SPBagInterface *>::Alloc;

#if CILKSAN_DEBUG
template<>
long DisjointSet_t<SPBagInterface *>::debug_count;
#endif

#endif // #ifndef _DISJOINTSET_H
