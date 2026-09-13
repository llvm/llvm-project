// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -verify %s

class RefCountable {
public:
  void ref() const;
  void deref() const;
};

void someFunction(RefCountable*);

namespace std {
inline namespace __1 {
namespace ranges {

template <typename Collection, typename Predicate>
bool any_of(Collection&& collection, Predicate&& predicate) { return true; }

namespace __all_of {
struct __fn {
  template <typename Collection, typename Predicate>
  constexpr bool operator()(const Collection& collection, Predicate predicate) const { return true; }
};
}
inline constexpr auto all_of = __all_of::__fn {};

}

template <typename Callback>
void other_function(Callback&& callback) { }

}
}

struct Collection { };

bool ranges_function_through_inline_namespace(RefCountable* obj, Collection& collection) {
  return std::ranges::any_of(collection, [obj](int) {
    someFunction(obj);
    return true;
  });
}

bool ranges_niebloid_through_inline_namespace(RefCountable* obj, Collection& collection) {
  return std::ranges::all_of(collection, [obj](int) {
    someFunction(obj);
    return true;
  });
}

void non_ranges_function_through_inline_namespace(RefCountable* obj) {
  std::other_function([obj] {
    // expected-warning@-1{{Captured variable 'obj' is a raw pointer to RefPtr-capable type 'RefCountable' [webkit.UncountedLambdaCapturesChecker]}}
    someFunction(obj);
  });
}
