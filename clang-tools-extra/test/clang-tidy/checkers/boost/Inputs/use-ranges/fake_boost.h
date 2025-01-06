#ifndef USE_RANGES_FAKE_BOOST_H
#define USE_RANGES_FAKE_BOOST_H

namespace boost {
namespace range_adl_barrier {

template <typename T> void *begin(T &);
template <typename T> void *end(T &);
template <typename T> void *const_begin(const T &);
template <typename T> void *const_end(const T &);
} // namespace range_adl_barrier

using namespace range_adl_barrier;

template <typename T> void *rbegin(T &);
template <typename T> void *rend(T &);

template <typename T> void *const_rbegin(T &);
template <typename T> void *const_rend(T &);
namespace algorithm {

template <class InputIterator, class T, class BinaryOperation>
T reduce(InputIterator first, InputIterator last, T init, BinaryOperation bOp) {
  return init;
}
} // namespace algorithm
} // namespace boost

#endif // USE_RANGES_FAKE_BOOST_H
