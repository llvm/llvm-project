namespace std {
  struct _Swallow_assign
  {
    template<class _Tp>
       const _Swallow_assign&
      operator=(const _Tp&) const
      { return *this; }
  };

  constexpr _Swallow_assign ignore{};

  template<typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;

    pair() = default;
    pair(T1 first, T2 second) : first(first), second(second) {}
  };

  template<typename... Args>
  struct tuple {
    tuple(Args&...) {}

    template<typename T1, typename T2>
    tuple<T1, T2> operator=(const std::pair<T1, T2>&);
  };

  template<typename... Args>
  tuple<Args...> tie(Args&... args) {
    return tuple<Args...>(args...);
  }

  template <typename Key, typename Value>
  class unordered_map {
  public:
    using value_type = pair<Key, Value>;

    class iterator {
    public:
      iterator& operator++();
      bool operator!=(const iterator &other);
      const value_type &operator*() const;
      value_type operator*();
      const value_type* operator->() const;
    };

    iterator begin() const;
    iterator end() const;
  };
}

template<typename T1, typename T2>
std::pair<T1, T2> getPair();

template<typename T1, typename T2>
constexpr std::pair<T1, T2> getConstexprPair() {
  return std::pair<T1, T2>();
}

template<typename T1, typename T2, typename T3>
std::tuple<T1, T2, T3> getTuple();
