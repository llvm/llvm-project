// Requires that T supports post and pre-incrementing.
template <typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};
