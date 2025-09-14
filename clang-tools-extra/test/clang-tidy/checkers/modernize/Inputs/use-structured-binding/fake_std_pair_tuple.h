namespace std {
  template<typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;
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
}

template<typename T1, typename T2>
std::pair<T1, T2> getPair();
