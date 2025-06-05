// RUN: %check_clang_tidy %s cppcoreguidelines-pro-bounds-pointer-arithmetic %t

namespace std {
template <typename, typename>
class pair {};

template <typename Key, typename Value>
class map {
  public:
   using value_type = pair<Key, Value>;
   value_type& operator[](const Key& key);
   value_type& operator[](Key&& key);
 };
}

template <typename R>
int f(std::map<R*, int>& map, R* r) {
  return map[r]; // OK
}
