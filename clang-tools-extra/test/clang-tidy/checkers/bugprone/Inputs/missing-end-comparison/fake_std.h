#ifndef MISSING_END_COMPARISON_FAKE_STD_H
#define MISSING_END_COMPARISON_FAKE_STD_H

namespace std {
  template<typename T> struct iterator_traits;
  struct forward_iterator_tag {};

  typedef long int ptrdiff_t;
  typedef decltype(nullptr) nullptr_t;

  template<typename T>
  struct vector {
    typedef T* iterator;
    typedef const T* const_iterator;
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
  };

  template<class InputIt, class T>
  InputIt find(InputIt first, InputIt last, const T& value);

  namespace execution {
    struct sequenced_policy {};
    struct parallel_policy {};
    inline constexpr sequenced_policy seq;
    inline constexpr parallel_policy par;
  }

  template<class ExecutionPolicy, class InputIt, class T>
  InputIt find(ExecutionPolicy&& policy, InputIt first, InputIt last, const T& value);

  template<class ExecutionPolicy, class ForwardIt, class T>
  ForwardIt lower_bound(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last, const T& value);

  template<class ForwardIt, class T>
  ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value);

  template<class ForwardIt, class ForwardIt2>
  ForwardIt search(ForwardIt first, ForwardIt last, ForwardIt first2, ForwardIt2 last2);

  template<class ForwardIt>
  ForwardIt min_element(ForwardIt first, ForwardIt last);

  template<class InputIt1, class InputIt2>
  struct pair {
    InputIt1 first;
    InputIt2 second;
  };

  namespace ranges {
    template<typename T>
    void* begin(T& t);
    template<typename T>
    void* end(T& t);

    struct FindFn {
      template<typename Range, typename T>
      void* operator()(Range&& r, const T& value) const;

      template<typename I, typename S, typename T>
      void* operator()(I first, S last, const T& value) const;
    };
    inline constexpr FindFn find;

    struct FindFirstOfFn {
      template<typename R1, typename R2>
      void* operator()(R1&& r1, R2&& r2) const;
      template<typename I1, typename S1, typename I2, typename S2>
      void* operator()(I1 f1, S1 l1, I2 f2, S2 l2) const;
    };
    inline constexpr FindFirstOfFn find_first_of;

    struct AdjacentFindFn {
      template<typename R>
      void* operator()(R&& r) const;
      template<typename I, typename S>
      void* operator()(I f, S l) const;
    };
    inline constexpr AdjacentFindFn adjacent_find;

    struct IsSortedUntilFn {
      template<typename R>
      void* operator()(R&& r) const;
      template<typename I, typename S>
      void* operator()(I f, S l) const;
    };
    inline constexpr IsSortedUntilFn is_sorted_until;
  }
} // namespace std

#endif // MISSING_END_COMPARISON_FAKE_STD_H
