namespace std {
  template<typename> struct mersenne_twister_engine {
    friend bool operator==(const mersenne_twister_engine &,
                           const mersenne_twister_engine &) {
      return false;
    }
  };
  struct random_device {
    mersenne_twister_engine<int> mt; // require complete type
  };
}

