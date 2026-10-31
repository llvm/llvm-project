// Requires that T suports post and pre-incrementing.
template <typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};
