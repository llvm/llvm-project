#ifndef SIZE_H
#define SIZE_H

namespace std {

template<class C> constexpr auto size(const C& c) noexcept(noexcept(c.size())) -> decltype(c.size());

} // namespace std

#endif  // SIZE_H
