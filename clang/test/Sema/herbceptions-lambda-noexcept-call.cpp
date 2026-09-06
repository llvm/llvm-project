// RUN: %clang_cc1 -std=c++26 -fherbceptions -fsyntax-only %s
// expected-no-diagnostics

namespace std {
template <class F, class... Args>
constexpr auto invoke(F&& f, Args&&... args) -> decltype(f((Args&&)args...)) {
  return f((Args&&)args...);
}
} // namespace std

struct Parent {
  int __fun_ = 0;
};

template <class _Parent>
struct __iterator {
  _Parent* __parent_ = nullptr;

  // Reproducer for a Sema::ActOnCallExpr null deref with -fherbceptions.
  // While the trailing exception specification of a lambda is being parsed,
  // the current function decl is the lambda's own call operator, which has
  // been created but does not have a type attached yet. The herbception
  // auto-propagation check in ActOnCallExpr dereferenced the null QualType
  // via getAs<FunctionProtoType>() and segfaulted. The call inside the
  // noexcept operand must be dependent, as in the libc++
  // ranges::zip_transform_view::__iterator::__get_deref_and_invoke pattern
  // that originally surfaced this.
  constexpr auto get() const noexcept {
    return [&__fun = *__parent_->__fun_](const auto&... __iters) noexcept(
               noexcept(std::invoke(*__parent_->__fun_, *__iters...)))
               -> decltype(auto) { return std::invoke(__fun, *__iters...); };
  }
};

int main() {
  __iterator<Parent> it;
  (void)it;
}
