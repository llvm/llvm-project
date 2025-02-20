#ifndef CONTAINERS_H
#define CONTAINERS_H

namespace std {

template <typename T>
class iterator {
public:
  iterator() {}
  iterator(const iterator<T> &iter) : ptr(iter.ptr) {}

  typedef T value_type;
  typedef T *pointer;
  typedef T &reference;

  reference operator*() const { return *ptr; }
  pointer operator->() const { return ptr; }
  iterator &operator++() {
    ++ptr;
    return *this;
  }
  iterator &operator--() {
    --ptr;
    return *this;
  }
  iterator operator++(int) {
    iterator res(*this);
    ++ptr;
    return res;
  }
  iterator operator--(int) {
    iterator res(*this);
    --ptr;
    return res;
  }
  bool operator!=(const iterator<T> &iter) const {
    return ptr != iter.operator->();
  }

private:
  T *ptr;
};

template <class Iterator>
class const_iterator {
public:
  const_iterator() {}
  const_iterator(const Iterator &iter) : iter(iter) {}
  const_iterator(const const_iterator<Iterator> &citer) : iter(citer.iter) {}

  typedef const typename Iterator::value_type value_type;
  typedef const typename Iterator::pointer pointer;
  typedef const typename Iterator::reference reference;

  reference operator*() const { return *iter; }
  pointer operator->() const { return iter.operator->(); }

  const_iterator &operator++() { return ++iter; }
  const_iterator &operator--() { return --iter; }
  const_iterator operator++(int) { return iter--; }
  const_iterator operator--(int) { return iter--; }

  bool operator!=(const Iterator &it) const {
    return iter->operator->() != it.operator->();
  }
  bool operator!=(const const_iterator<Iterator> &it) const {
    return iter.operator->() != it.operator->();
  }

private:
  Iterator iter;
};

template <class Iterator>
class forward_iterable {
public:
  forward_iterable() {}
  typedef Iterator iterator;
  typedef const_iterator<Iterator> const_iterator;

  iterator begin() { return _begin; }
  iterator end() { return _end; }

  const_iterator begin() const { return _begin; }
  const_iterator end() const { return _end; }

  const_iterator cbegin() const { return _begin; }
  const_iterator cend() const { return _end; }

private:
  iterator _begin, _end;
};

template <class Iterator>
class reverse_iterator {
public:
  reverse_iterator() {}
  reverse_iterator(const Iterator &iter) : iter(iter) {}
  reverse_iterator(const reverse_iterator<Iterator> &rit) : iter(rit.iter) {}

  typedef typename Iterator::value_type value_type;
  typedef typename Iterator::pointer pointer;
  typedef typename Iterator::reference reference;

  reference operator*() { return *iter; }
  pointer operator->() { return iter.operator->(); }

  reverse_iterator &operator++() { return --iter; }
  reverse_iterator &operator--() { return ++iter; }
  reverse_iterator operator++(int) { return iter--; }
  reverse_iterator operator--(int) { return iter++; }

private:
  Iterator iter;
};

template <class Iterator>
class backward_iterable {
public:
  backward_iterable() {}

  typedef reverse_iterator<Iterator> reverse_iterator;
  typedef const_iterator<reverse_iterator> const_reverse_iterator;

  reverse_iterator rbegin() { return _rbegin; }
  reverse_iterator rend() { return _rend; }

  const_reverse_iterator rbegin() const { return _rbegin; }
  const_reverse_iterator rend() const { return _rend; }

  const_reverse_iterator crbegin() const { return _rbegin; }
  const_reverse_iterator crend() const { return _rend; }

private:
  reverse_iterator _rbegin, _rend;
};

template <class Iterator>
class bidirectional_iterable : public forward_iterable<Iterator>,
                               public backward_iterable<Iterator> {};

template <typename A, typename B>
struct pair {
  pair(A f, B s) : first(f), second(s) {}
  A first;
  B second;
};

class string {
public:
  string() {}
  string(const char *) {}
};

template <typename T, int n>
class array : public backward_iterable<iterator<T>> {
public:
  array() {}

  typedef T *iterator;
  typedef const T *const_iterator;

  iterator begin() { return &v[0]; }
  iterator end() { return &v[n - 1]; }

  const_iterator begin() const { return &v[0]; }
  const_iterator end() const { return &v[n - 1]; }

  const_iterator cbegin() const { return &v[0]; }
  const_iterator cend() const { return &v[n - 1]; }

private:
  T v[n];
};

template <typename T>
class deque : public bidirectional_iterable<iterator<T>> {
public:
  deque() {}
};

template <typename T>
class list : public bidirectional_iterable<iterator<T>> {
public:
  list() {}
};

template <typename T>
class forward_list : public forward_iterable<iterator<T>> {
public:
  forward_list() {}
};

template <typename T>
class vector : public bidirectional_iterable<iterator<T>> {
public:
  vector() {}
};

template <typename T>
class set : public bidirectional_iterable<iterator<T>> {
public:
  set() {}
};

template <typename T>
class multiset : public bidirectional_iterable<iterator<T>> {
public:
  multiset() {}
};

template <typename key, typename value>
class map : public bidirectional_iterable<iterator<pair<key, value>>> {
public:
  map() {}

  iterator<pair<key, value>> find(const key &) {}
  const_iterator<iterator<pair<key, value>>> find(const key &) const {}
};

template <typename key, typename value>
class multimap : public bidirectional_iterable<iterator<pair<key, value>>> {
public:
  multimap() {}
};

template <typename T>
class unordered_set : public forward_iterable<iterator<T>> {
public:
  unordered_set() {}
};

template <typename T>
class unordered_multiset : public forward_iterable<iterator<T>> {
public:
  unordered_multiset() {}
};

template <typename key, typename value>
class unordered_map : public forward_iterable<iterator<pair<key, value>>> {
public:
  unordered_map() {}
};

template <typename key, typename value>
class unordered_multimap : public forward_iterable<iterator<pair<key, value>>> {
public:
  unordered_multimap() {}
};

} // namespace std

#endif // CONTAINERS_H
