//Run %clang_cc1 -std=c++20 -verify %s
godbolt — clang trunk / 22.1.0 / 21.1.0 / gcc 15.1.0 side-by-side.

#include <concepts>
#include <type_traits>

struct base_marker {};

template <typename Tag>
struct outer {
  struct inner : base_marker {
    template <typename L, typename S>
      requires std::derived_from<std::remove_cvref_t<L>, base_marker> &&
               std::same_as<std::remove_cvref_t<S>, inner>
    friend void operator|(L&&, S&&) {}
  };
};

int main() {
  typename outer<int>::inner a;
  typename outer<double>::inner b;
  a | b;
}

if (Function->getFriendObjectKind()) {
  if (ForConstraintInstantiation)
    return Response::UseNextDecl(Function);

  return Response::ChangeDecl(Function->getLexicalDeclContext());
}