// RUN: %check_clang_tidy %s performance-expensive-flat-container-operation %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: performance-expensive-flat-container-operation.WarnOutsideLoops, \
// RUN:               value: true}] \
// RUN:             }"

#include <stddef.h>

namespace std {
template <class T1, class T2> struct pair { pair(T1, T2); };

template <class T> struct initializer_list {};

template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T &> { typedef T type; };
template <class T> struct remove_reference<T &&> { typedef T type; };

template <class T>
typename std::remove_reference<T>::type &&move(T &&) noexcept;

struct sorted_unique_t {};
inline constexpr sorted_unique_t sorted_unique{};
struct sorted_equivalent_t {};
inline constexpr sorted_equivalent_t sorted_equivalent{};

template <class Key, class T> struct flat_map {
  using key_type = Key;
  using mapped_type = T;
  using value_type = pair<key_type, mapped_type>;
  using reference = pair<const key_type &, mapped_type &>;
  using const_reference = pair<const key_type &, const mapped_type &>;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <class... Args> pair<iterator, bool> emplace(Args &&...args);
  template <class... Args>
  iterator emplace_hint(const_iterator position, Args &&...args);

  pair<iterator, bool> insert(const value_type &x);
  pair<iterator, bool> insert(value_type &&x);
  iterator insert(const_iterator position, const value_type &x);
  iterator insert(const_iterator position, value_type &&x);

  // template <class P> pair<iterator, bool> insert(P &&x);
  // template <class P> iterator insert(const_iterator position, P &&);
  template <class InputIter> void insert(InputIter first, InputIter last);
  template <class InputIter>
  void insert(sorted_unique_t, InputIter first, InputIter last);
  template <class R> void insert_range(R &&rg);

  void insert(initializer_list<value_type> il);
  void insert(sorted_unique_t s, initializer_list<value_type> il);

  template <class... Args>
  pair<iterator, bool> try_emplace(const key_type &k, Args &&...args);
  template <class... Args>
  pair<iterator, bool> try_emplace(key_type &&k, Args &&...args);
  template <class K, class... Args>
  pair<iterator, bool> try_emplace(K &&k, Args &&...args);
  template <class... Args>
  iterator try_emplace(const_iterator hint, const key_type &k, Args &&...args);
  template <class... Args>
  iterator try_emplace(const_iterator hint, key_type &&k, Args &&...args);
  template <class K, class... Args>
  iterator try_emplace(const_iterator hint, K &&k, Args &&...args);
  template <class M>
  pair<iterator, bool> insert_or_assign(const key_type &k, M &&obj);
  template <class M>
  pair<iterator, bool> insert_or_assign(key_type &&k, M &&obj);
  template <class K, class M>
  pair<iterator, bool> insert_or_assign(K &&k, M &&obj);
  template <class M>
  iterator insert_or_assign(const_iterator hint, const key_type &k, M &&obj);
  template <class M>
  iterator insert_or_assign(const_iterator hint, key_type &&k, M &&obj);
  template <class K, class M>
  iterator insert_or_assign(const_iterator hint, K &&k, M &&obj);

  iterator erase(iterator position);
  iterator erase(const_iterator position);
  size_type erase(const key_type &x);
  template <class K> size_type erase(K &&x);
  iterator erase(const_iterator first, const_iterator last);
};

template <class Key, class T> struct flat_multimap {
  using key_type = Key;
  using mapped_type = T;
  using value_type = pair<key_type, mapped_type>;
  using reference = pair<const key_type &, mapped_type &>;
  using const_reference = pair<const key_type &, const mapped_type &>;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <class... Args> iterator emplace(Args &&...args);
  template <class... Args>
  iterator emplace_hint(const_iterator position, Args &&...args);

  iterator insert(const value_type &x);
  iterator insert(value_type &&x);
  iterator insert(const_iterator position, const value_type &x);
  iterator insert(const_iterator position, value_type &&x);

  // template <class P> iterator insert(P &&x);
  // template <class P> iterator insert(const_iterator position, P &&);
  template <class InputIter> void insert(InputIter first, InputIter last);
  template <class InputIter>
  void insert(sorted_equivalent_t, InputIter first, InputIter last);
  template <class R> void insert_range(R &&rg);

  void insert(initializer_list<value_type> il);
  void insert(sorted_equivalent_t s, initializer_list<value_type> il);

  iterator erase(iterator position);
  iterator erase(const_iterator position);
  size_type erase(const key_type &x);
  template <class K> size_type erase(K &&x);
  iterator erase(const_iterator first, const_iterator last);

  void swap(flat_multimap &) noexcept;
  void clear() noexcept;
};

template <class Key> struct flat_set {
  using key_type = Key;
  using value_type = Key;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <class... Args> pair<iterator, bool> emplace(Args &&...args);
  template <class... Args>
  iterator emplace_hint(const_iterator position, Args &&...args);

  pair<iterator, bool> insert(const value_type &x);
  pair<iterator, bool> insert(value_type &&x);
  template <class K> pair<iterator, bool> insert(K &&x);
  iterator insert(const_iterator position, const value_type &x);
  iterator insert(const_iterator position, value_type &&x);
  template <class K> iterator insert(const_iterator hint, K &&x);

  template <class InputIter> void insert(InputIter first, InputIter last);
  template <class InputIter>
  void insert(sorted_unique_t, InputIter first, InputIter last);
  template <class R> void insert_range(R &&rg);

  void insert(initializer_list<value_type> il);
  void insert(sorted_unique_t s, initializer_list<value_type> il);

  iterator erase(iterator position);
  iterator erase(const_iterator position);
  size_type erase(const key_type &x);
  template <class K> size_type erase(K &&x);
  iterator erase(const_iterator first, const_iterator last);
};

template <class Key> struct flat_multiset {
  using key_type = Key;
  using value_type = Key;
  using reference = value_type &;
  using const_reference = const value_type &;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <class... Args> iterator emplace(Args &&...args);
  template <class... Args>
  iterator emplace_hint(const_iterator position, Args &&...args);

  iterator insert(const value_type &x);
  iterator insert(value_type &&x);
  iterator insert(const_iterator position, const value_type &x);
  iterator insert(const_iterator position, value_type &&x);

  template <class InputIter> void insert(InputIter first, InputIter last);
  template <class InputIter>
  void insert(sorted_equivalent_t, InputIter first, InputIter last);
  template <class R> void insert_range(R &&rg);

  void insert(initializer_list<value_type> il);
  void insert(sorted_equivalent_t s, initializer_list<value_type> il);

  iterator erase(iterator position);
  iterator erase(const_iterator position);
  size_type erase(const key_type &x);
  template <class K> size_type erase(K &&x);
  iterator erase(const_iterator first, const_iterator last);
};
} // namespace std

namespace boost::container {
struct ordered_unique_range_t {};
inline constexpr ordered_unique_range_t ordered_unique_range{};
struct ordered_range_t {};
inline constexpr ordered_range_t ordered_range{};

template <typename Key, typename T> struct flat_map {
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<Key, T>;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <typename M>
  std::pair<iterator, bool> insert_or_assign(const key_type &, M &&);
  template <typename M>
  std::pair<iterator, bool> insert_or_assign(key_type &&, M &&);
  template <typename M>
  iterator insert_or_assign(const_iterator, const key_type &, M &&);
  template <typename M>
  iterator insert_or_assign(const_iterator, key_type &&, M &&);
  template <class... Args> std::pair<iterator, bool> emplace(Args &&...);
  template <class... Args> iterator emplace_hint(const_iterator, Args &&...);
  template <class... Args>
  std::pair<iterator, bool> try_emplace(const key_type &, Args &&...);
  template <class... Args>
  iterator try_emplace(const_iterator, const key_type &, Args &&...);
  template <class... Args>
  std::pair<iterator, bool> try_emplace(key_type &&, Args &&...);
  template <class... Args>
  iterator try_emplace(const_iterator, key_type &&, Args &&...);
  std::pair<iterator, bool> insert(const value_type &);
  std::pair<iterator, bool> insert(value_type &&);
  template <typename Pair> std::pair<iterator, bool> insert(Pair &&);
  iterator insert(const_iterator, const value_type &);
  iterator insert(const_iterator, value_type &&);
  template <typename Pair> iterator insert(const_iterator, Pair &&);
  template <typename InputIterator> void insert(InputIterator, InputIterator);
  template <typename InputIterator>
  void insert(ordered_unique_range_t, InputIterator, InputIterator);
  void insert(std::initializer_list<value_type>);
  void insert(ordered_unique_range_t, std::initializer_list<value_type>);
  iterator erase(const_iterator);
  size_type erase(const key_type &);
  iterator erase(const_iterator, const_iterator);
};

template <typename Key, typename T> struct flat_multimap {
  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<Key, T>;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  template <class... Args> iterator emplace(Args &&...);
  template <class... Args> iterator emplace_hint(const_iterator, Args &&...);
  iterator insert(const value_type &);
  template <typename Pair> iterator insert(Pair &&);
  iterator insert(const_iterator, const value_type &);
  template <typename Pair> iterator insert(const_iterator, Pair &&);
  template <typename InputIterator> void insert(InputIterator, InputIterator);
  template <typename InputIterator>
  void insert(ordered_range_t, InputIterator, InputIterator);
  void insert(std::initializer_list<value_type>);
  void insert(ordered_range_t, std::initializer_list<value_type>);
  iterator erase(const_iterator);
  size_type erase(const key_type &);
  iterator erase(const_iterator, const_iterator);
};

template <typename Key> struct flat_set {
  using key_type = Key;
  using value_type = Key;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  template <class... Args> std::pair<iterator, bool> emplace(Args &&...);
  template <class... Args> iterator emplace_hint(const_iterator, Args &&...);
  std::pair<iterator, bool> insert(const value_type &);
  std::pair<iterator, bool> insert(value_type &&);
  iterator insert(const_iterator, const value_type &);
  iterator insert(const_iterator, value_type &&);
  template <typename InputIterator> void insert(InputIterator, InputIterator);
  template <typename InputIterator>
  void insert(ordered_unique_range_t, InputIterator, InputIterator);
  void insert(std::initializer_list<value_type>);
  void insert(ordered_unique_range_t, std::initializer_list<value_type>);
  size_type erase(const key_type &);
  iterator erase(const_iterator);
  iterator erase(const_iterator, const_iterator);
};

template <typename Key> struct flat_multiset {
  using key_type = Key;
  using value_type = Key;
  using size_type = size_t;
  using iterator = struct {};
  using const_iterator = struct {};

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;
  template <class... Args> iterator emplace(Args &&...);
  template <class... Args> iterator emplace_hint(const_iterator, Args &&...);
  iterator insert(const value_type &);
  iterator insert(value_type &&);
  iterator insert(const_iterator, const value_type &);
  iterator insert(const_iterator, value_type &&);
  template <typename InputIterator> void insert(InputIterator, InputIterator);
  template <typename InputIterator>
  void insert(ordered_range_t, InputIterator, InputIterator);
  void insert(std::initializer_list<value_type>);
  void insert(ordered_range_t, std::initializer_list<value_type>);
  iterator erase(const_iterator);
  size_type erase(const key_type &);
  iterator erase(const_iterator, const_iterator);
};
} // namespace boost::container

namespace folly {
struct sorted_unique_t {};
inline constexpr sorted_unique_t sorted_unique{};

template <class T> struct sorted_vector_set {
  using value_type = T;
  using key_type = T;
  using iterator = struct {};
  using const_iterator = struct {};
  using size_type = size_t;

  const_iterator begin() const;
  const_iterator end() const;

  std::pair<iterator, bool> insert(const value_type &value);
  std::pair<iterator, bool> insert(value_type &&value);
  iterator insert(const_iterator hint, const value_type &value);
  iterator insert(const_iterator hint, value_type &&value);
  template <class InputIterator>
  void insert(InputIterator first, InputIterator last);
  template <class InputIterator>
  void insert(sorted_unique_t, InputIterator first, InputIterator last);
  void insert(std::initializer_list<value_type> ilist);

  template <typename... Args> std::pair<iterator, bool> emplace(Args &&...args);
  std::pair<iterator, bool> emplace(const value_type &value);
  std::pair<iterator, bool> emplace(value_type &&value);

  template <typename... Args>
  iterator emplace_hint(const_iterator hint, Args &&...args);
  iterator emplace_hint(const_iterator hint, const value_type &value);
  iterator emplace_hint(const_iterator hint, value_type &&value);

  size_type erase(const key_type &key);
  iterator erase(const_iterator it);
  iterator erase(const_iterator first, const_iterator last);
};

template <class Key, class Value> struct sorted_vector_map {
  typedef Key key_type;
  typedef Value mapped_type;
  using value_type = std::pair<Key, Value>;
  using iterator = struct {};
  using const_iterator = struct {};
  using size_type = size_t;

  const_iterator begin() const;
  const_iterator end() const;

  std::pair<iterator, bool> insert(const value_type &value);
  std::pair<iterator, bool> insert(value_type &&value);
  iterator insert(const_iterator hint, const value_type &value);
  iterator insert(const_iterator hint, value_type &&value);
  template <class InputIterator>
  void insert(InputIterator first, InputIterator last);
  template <class InputIterator>
  void insert(sorted_unique_t, InputIterator first, InputIterator last);
  void insert(std::initializer_list<value_type> ilist);

  template <typename... Args> std::pair<iterator, bool> emplace(Args &&...args);
  std::pair<iterator, bool> emplace(const value_type &value);
  std::pair<iterator, bool> emplace(value_type &&value);

  template <typename... Args>
  iterator emplace_hint(const_iterator hint, Args &&...args);
  iterator emplace_hint(const_iterator hint, const value_type &value);
  iterator emplace_hint(const_iterator hint, value_type &&value);

  template <typename... Args>
  std::pair<iterator, bool> try_emplace(key_type &&k, Args &&...args);
  template <typename... Args>
  std::pair<iterator, bool> try_emplace(const key_type &k, Args &&...args);

  template <typename M>
  std::pair<iterator, bool> insert_or_assign(const key_type &k, M &&obj);
  template <typename M>
  std::pair<iterator, bool> insert_or_assign(key_type &&k, M &&obj);
  template <class M>
  iterator insert_or_assign(const_iterator hint, const key_type &k, M &&obj);
  template <class M>
  iterator insert_or_assign(const_iterator hint, key_type &&k, M &&obj);

  size_type erase(const key_type &key);
  iterator erase(const_iterator it);
  iterator erase(const_iterator first, const_iterator last);
};
} // namespace folly

struct MyKey {};

void testStdFlapMapErase(std::flat_map<MyKey, int> map) {
  map.erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.

  map.erase(map.begin());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.

  map.erase(map.begin(), map.end());
}

void testWithUsing() {
  using MySpecialMap = std::flat_map<MyKey, int>;
  MySpecialMap map;
  map.erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.
}

void testWithArrow() {
  auto *map = new std::flat_map<MyKey, int>();
  map->erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.
}

void testWithSubclass() {
  struct MyMap : std::flat_map<MyKey, int> {};
  MyMap map;
  map.erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.

  using MySpecialMap = MyMap;
  MySpecialMap map2;
  map2.erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.

  using MySpecialMap2 = std::flat_map<MyKey, int>;
  struct MyMap2 : MySpecialMap2 {};
  MyMap2 map3;
  map3.erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.

  using MySpecialMap3 = MyMap2;
  auto *map4 = new MySpecialMap3();
  map4->erase(MyKey());
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
  // operations are expensive for flat containers.
}
