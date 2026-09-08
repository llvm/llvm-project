#pragma once
namespace std {
struct error_domain_singleton {
  void (*do_cleanup)(unsigned long) noexcept = 0;
  bool (*do_equivalent)(unsigned long, error_domain_singleton const *,
                        unsigned long) noexcept = 0;
  void (*do_query_information)(unsigned long, int, int, void *, void *) noexcept = 0;
  int (*do_to_errc)(unsigned long) noexcept = 0;
  void (*do_throw_dynamic_exception)(unsigned long, void *) = 0;
};
class error {
public:
  error() = delete;
  error(error const &) = delete;
  error &operator=(error const &) = delete;
  constexpr ~error() noexcept {}
  [[nodiscard]] unsigned long code() const noexcept { return c; }
private:
  void const *d{};
  unsigned long c{};
};
template <typename T> struct error_domain;
enum class errc : int { io_error = 5 };
inline const error_domain_singleton herb_dom{};
template <> struct error_domain<errc> {
  static constexpr error_domain_singleton const *domain() noexcept {
    return &herb_dom;
  }
  static constexpr unsigned long code(errc e) noexcept {
    return static_cast<unsigned long>(e);
  }
};
} // namespace std

// Inline definitions with bodies: these force FunctionDecl definitions to be
// serialized into the module file (the path that regressed when the legacy
// conversion expression moved before the body block).
inline void throws_io() throws { throw throws std::errc::io_error; }
inline int fail_fn(int x) return_failure{int} {
  if (x < 0)
    return_failure x;
  return x;
}
