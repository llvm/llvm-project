// RUN: %check_clang_tidy %s performance-use-std-move %t -- -config='{CheckOptions: {performance-use-std-move.AllowedTypes: "Ignored;::view::.*"}}'
// RUN: %check_clang_tidy %s performance-use-std-move %t -- -config='{CheckOptions: {performance-use-std-move.AllowedTypes: "Ignored;::view::.*", performance-use-std-move.IncludeStyle: google}}'
// RUN: %clang -std=c++11 -fsyntax-only -nostdinc++ -isystem %clang_tidy_headers/std %t.cpp

// CHECK-FIXES: #include <utility>

struct Movable {
  Movable();
  Movable(const Movable &);
  Movable(Movable &&);
  Movable &operator=(const Movable &);
  Movable &operator=(Movable &&);
};
struct Ignored {
  Ignored();
  Ignored(const Ignored &);
  Ignored(Ignored &&);
  Ignored &operator=(const Ignored &);
  Ignored &operator=(Ignored &&);
};
namespace view {
struct CustomView {
  CustomView();
  CustomView(const CustomView &);
  CustomView(CustomView &&);
};
} // namespace view

void insertUtility() {
  Movable source;
  Movable target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Movable target(std::move(source));
}
void excludedCopy(Ignored source) {
  Ignored target(source);
  // CHECK-FIXES: Ignored target(source);
}
void excludedAssignment(Ignored &target, Ignored source) {
  target = source;
  // CHECK-FIXES: target = source;
}
void excludedView(view::CustomView source) {
  view::CustomView target(source);
  // CHECK-FIXES: view::CustomView target(source);
}
void insertUtilityAssignment(Movable &target, Movable source) {
  target = source;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: target = std::move(source);
}
