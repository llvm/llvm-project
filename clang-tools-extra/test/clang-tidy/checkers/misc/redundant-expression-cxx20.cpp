// RUN: clang-tidy %s -checks=-*,misc-redundant-expression -- -std=c++20 | count 0

namespace concepts {
// redundant expressions inside concepts make sense, ignore them
template <class I>
concept TestConcept = requires(I i) {
  {i - i};
  {i && i};
  {i ? i : i};
};
} // namespace concepts
