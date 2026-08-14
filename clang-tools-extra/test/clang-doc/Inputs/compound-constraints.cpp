/// \brief Concept for an incrementable value
///
/// \tparam T A value that can be incremented.
template <typename T>
concept Incrementable = requires(T a) { a++; };

/// \brief Concept for a decrementable value
///
/// \tparam T A value that can be decremented
template <typename T>
concept Decrementable = requires(T a) { a--; };

/// \brief Concept for a pre-incrementable value
///
/// \tparam T A value that can be pre-incremented
template <typename T>
concept PreIncrementable = requires(T a) { ++a; };

/// \brief Concept for a -pre-decrementable value
///
/// \tparam T A value that can be pre-decremented
template <typename T>
concept PreDecrementable = requires(T a) { --a; };

template <typename T>
  requires Incrementable<T> && Decrementable<T>
void One();

template <typename T>
  requires(Incrementable<T> && Decrementable<T>)
void Two();

template <typename T>
  requires(Incrementable<T> && Decrementable<T>) ||
          (PreIncrementable<T> && PreDecrementable<T>)
void Three();

template <typename T>
  requires(Incrementable<T> && Decrementable<T>) || PreIncrementable<T>
void Four();
