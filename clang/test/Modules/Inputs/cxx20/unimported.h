namespace StructuredBinding {
  struct Q { int p, q; };
  static auto [a, b] = Q();
}
