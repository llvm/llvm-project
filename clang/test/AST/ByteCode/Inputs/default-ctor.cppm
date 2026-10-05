export module m;

export struct allocator_like {
  constexpr allocator_like() noexcept = default;
};

export struct box {
  allocator_like allocation;
  int value = 42;
  constexpr box() = default;
  [[nodiscard]] constexpr int get() const { return value; }
};
