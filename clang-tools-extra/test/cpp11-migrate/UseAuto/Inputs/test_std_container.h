//===-----------------------------------------------------------*- C++ -*--===//
//
// This file contains a shell implementation of a standard container with
// iterators. This shell is targeted at supporting the container interfaces
// recognized by cpp11-migrate's use-auto transformation. It requires the
// preprocessor to parameterize the name of the container, and allows the
// preprocessor to parameterize various mechanisms used in the implementation
// of the container / iterator.
//
// Variations for how iterator types are presented:
// * Typedef (array, deque, forward_list, list, vector)
// * Nested class (map, multimap, set, multiset)
// * Using declaration {unordered_} X {map, multimap, set, multiset}
//
// Variations for how container types are presented:
// * Defined directly in namespace std
// * Imported into namespace std with using declarations (a la libc++).
//
//===----------------------------------------------------------------------===//

#ifndef CONTAINER
#error You must define CONTAINER to the name of the desired container.
#endif

// If the test code needs multiple containers, only define our helpers once.
#ifndef TEST_STD_CONTAINER_HELPERS
#define TEST_STD_CONTAINER_HELPERS

namespace internal {

template <typename T, int i>
struct iterator_wrapper {
  iterator_wrapper() {}

  // These are required for tests using iteration statements.
  bool operator!=(const iterator_wrapper<T, i>&) { return false; }
  iterator_wrapper& operator++() { return *this; }
  typename T::value_type operator*() { return typename T::value_type(); }
};

template <typename T>
class iterator_provider {
public:
  class iterator {
  public:
    iterator() {}
    iterator(const iterator&) {}
  };
  class const_iterator {
  public:
    const_iterator(int i=0) {}
    const_iterator(const iterator &) {}
    const_iterator(const const_iterator &) {}
    operator iterator() { return iterator(); }
  };
  class reverse_iterator {};
  class const_reverse_iterator {};
};

} // namespace internal

#endif // TEST_STD_CONTAINER_HELPERS

namespace std {

#if USE_INLINE_NAMESPACE
namespace _1 {
#endif

template <typename T>
class CONTAINER
#if USE_BASE_CLASS_ITERATORS
  : internal::iterator_provider<CONTAINER<T> >
#endif
{
public:

#if USE_BASE_CLASS_ITERATORS
  using typename internal::iterator_provider<CONTAINER<T> >::iterator;
  using typename internal::iterator_provider<CONTAINER<T> >::const_iterator;
  using typename internal::iterator_provider<CONTAINER<T> >::reverse_iterator;
  using typename internal::iterator_provider<CONTAINER<T> >::const_reverse_iterator;
#elif USE_INNER_CLASS_ITERATORS
  class iterator {};
  class const_iterator {};
  class reverse_iterator {};
  class const_reverse_iterator {};
#else
  typedef T value_type;
  typedef typename internal::iterator_wrapper<CONTAINER<T>, 0> iterator;
  typedef typename internal::iterator_wrapper<CONTAINER<T>, 1> const_iterator;
  typedef typename internal::iterator_wrapper<CONTAINER<T>, 3> reverse_iterator;
  typedef typename internal::iterator_wrapper<CONTAINER<T>, 2> const_reverse_iterator;
#endif

  // Every class requires these functions.
  CONTAINER() {}

  iterator begin() { return iterator(); }
  iterator end() { return iterator(); }

  const_iterator begin() const { return const_iterator(); }
  const_iterator end() const { return const_iterator(); }

  reverse_iterator rbegin() { return reverse_iterator(); }
  reverse_iterator rend() { return reverse_iterator(); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(); }
  const_reverse_iterator rend() const { return const_reverse_iterator(); }

  template <typename K>
  iterator find(const K &Key) { return iterator(); }
};

#if USE_INLINE_NAMESPACE
} // namespace _1
using _1::CONTAINER;
#endif

} // namespace std
