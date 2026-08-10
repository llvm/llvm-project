#pragma once

namespace std {
template <class InputIterator, class T>
T accumulate(InputIterator first, InputIterator last, T init) {
  for (; first != last; ++first)
    init = init + *first;
  return init;
}
} // namespace std
