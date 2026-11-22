// RUN: %check_clang_tidy -std=c++14-or-later %s misc-const-correctness %t -- \
// RUN:   -- -fno-delayed-template-parsing

namespace gh235460 {
// A variable in a generic lambda that is only modified in a pack expansion
// must not be diagnosed, even though the instantiation with an empty pack
// does not modify it.
void modified_in_pack_expansion() {
  const auto sum = [](auto const &...xs) {
    unsigned long n = 0;
    using expand = int[];
    (void)expand{0, (n += sizeof(xs), 0)...};
    return n;
  };
  (void)sum();
  (void)sum(1);
}
} // namespace gh235460
