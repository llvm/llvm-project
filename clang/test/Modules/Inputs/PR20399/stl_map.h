namespace std {
struct reverse_iterator {};

inline void
operator-(int __x, reverse_iterator __y) {}

template <typename _Key>
struct map {
  typedef int iterator;

  friend bool operator<(const map &, const map &);
};
} // namespace std
