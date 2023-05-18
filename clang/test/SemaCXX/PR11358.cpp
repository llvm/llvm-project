// RUN: %clang_cc1 %s -verify
// PR11358

namespace test1 {
  template<typename T>
  struct container {
    class iterator {};
    iterator begin() { return iterator(); }
  };

  template<typename T>
  struct Test {
    typedef container<T> Container;
    void test() {
      Container::iterator i = c.begin(); // expected-error{{missing 'typename'}}
    }
    Container c;
  };
}

namespace test2 {
  template <typename Key, typename Value>
  class hash_map {
    class const_iterator { void operator++(); };
    const_iterator begin() const;
    const_iterator end() const;
  };

  template <typename KeyType, typename ValueType>
  void MapTest(hash_map<KeyType, ValueType> map) {
    for (hash_map<KeyType, ValueType>::const_iterator it = map.begin(); // expected-error{{missing 'typename'}}
         it != map.end(); it++) {
    }
  }
}

namespace test3 {
  template<typename T>
  struct container {
    class iterator {};
  };

  template<typename T>
  struct Test {
    typedef container<T> Container;
    void test() {
      Container::iterator const i; // expected-error{{missing 'typename'}}
    }
    Container c;
  };
}
