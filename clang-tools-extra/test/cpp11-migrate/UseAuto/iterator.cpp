// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -use-auto %t.cpp -- --std=c++11 -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#define CONTAINER array
#include "test_std_container.h"
#undef CONTAINER

#define CONTAINER vector
#include "test_std_container.h"
#undef CONTAINER

#define CONTAINER unordered_map
#define USE_BASE_CLASS_ITERATORS 1
#include "test_std_container.h"
#undef USE_BASE_CLASS_ITERATORS
#undef CONTAINER

typedef std::vector<int>::iterator int_iterator;

namespace foo {
  template <typename T>
  class vector {
  public:
    class iterator {};

    iterator begin() { return iterator(); }
  };
} // namespace foo

int main(int argc, char **argv) {
  std::vector<int> Vec;
  // CHECK: std::vector<int> Vec;

  std::unordered_map<int> Map;
  // CHECK: std::unordered_map<int> Map;

  // Types with more sugar should work. Types with less should not.
  {
    int_iterator more_sugar = Vec.begin();
    // CHECK: auto more_sugar = Vec.begin();

    internal::iterator_wrapper<std::vector<int>, 0> less_sugar = Vec.begin();
    // CHECK: internal::iterator_wrapper<std::vector<int>, 0> less_sugar = Vec.begin();
  }

  // Initialization from initializer lists isn't allowed. Using 'auto'
  // would result in std::initializer_list being deduced for the type.
  {
    std::unordered_map<int>::iterator I{Map.begin()};
    // CHECK: std::unordered_map<int>::iterator I{Map.begin()};

    std::unordered_map<int>::iterator I2 = {Map.begin()};
    // CHECK: std::unordered_map<int>::iterator I2 = {Map.begin()};
  }

  // Various forms of construction. Default constructors and constructors with
  // all-default parameters shouldn't get transformed. Construction from other
  // types is also not allowed.
  {
    std::unordered_map<int>::iterator copy(Map.begin());
    // CHECK: auto copy(Map.begin());

    std::unordered_map<int>::iterator def;
    // CHECK: std::unordered_map<int>::iterator def;

    // const_iterator has no default constructor, just one that has >0 params
    // with defaults.
    std::unordered_map<int>::const_iterator constI;
    // CHECK: std::unordered_map<int>::const_iterator constI;

    // Uses iterator_provider::const_iterator's conversion constructor.

    std::unordered_map<int>::const_iterator constI2 = def;
    // CHECK: std::unordered_map<int>::const_iterator constI2 = def;

    std::unordered_map<int>::const_iterator constI3(def);
    // CHECK: std::unordered_map<int>::const_iterator constI3(def);

    // Explicit use of conversion constructor

    std::unordered_map<int>::const_iterator constI4 = std::unordered_map<int>::const_iterator(def);
    // CHECK: auto constI4 = std::unordered_map<int>::const_iterator(def);

    // Uses iterator_provider::iterator's const_iterator conversion operator.

    std::unordered_map<int>::iterator I = constI;
    // CHECK: std::unordered_map<int>::iterator I = constI;

    std::unordered_map<int>::iterator I2(constI);
    // CHECK: std::unordered_map<int>::iterator I2(constI);
  }

  // Weird cases of pointers and references to iterators are not transformed.
  {
    int_iterator I = Vec.begin();

    int_iterator *IPtr = &I;
    // CHECK: int_iterator *IPtr = &I;

    int_iterator &IRef = I;
    // CHECK: int_iterator &IRef = I;
  }

  {
    // Variable declarations in iteration statements.
    for (std::vector<int>::iterator I = Vec.begin(); I != Vec.end(); ++I) {
      // CHECK: for (auto I = Vec.begin(); I != Vec.end(); ++I) {
    }

    // Range-based for loops.
    std::array<std::vector<int>::iterator> iter_arr;
    for (std::vector<int>::iterator I: iter_arr) {
      // CHECK: for (auto I: iter_arr) {
    }

    // Test with init-declarator-list.
    for (int_iterator I = Vec.begin(),
         E = Vec.end(); I != E; ++I) {
      // CHECK:      for (auto I = Vec.begin(),
      // CHECK-NEXT:      E = Vec.end(); I != E; ++I) {
    }
  }

  // Only std containers should be changed.
  {
    using namespace foo;
    vector<int> foo_vec;
    vector<int>::iterator I = foo_vec.begin();
    // CHECK: vector<int>::iterator I = foo_vec.begin();
  }

  // Ensure using directives don't interfere with replacement.
  {
    using namespace std;
    vector<int> std_vec;
    vector<int>::iterator I = std_vec.begin();
    // CHECK: auto I = std_vec.begin();
  }

  // Make sure references and cv qualifiers don't get removed (i.e. replaced
  // with just 'auto').
  {
    const auto & I = Vec.begin();
    // CHECK: const auto & I = Vec.begin();

    auto && I2 = Vec.begin();
    // CHECK: auto && I2 = Vec.begin();
  }

  // Passing a string as an argument to introduce a temporary object
  // that will create an expression with cleanups. Bugzilla: 15550
  {
    std::unordered_map<int> MapFind;
    std::unordered_map<int>::iterator I = MapFind.find("foo");
    // CHECK: auto I = MapFind.find("foo");
  }

  return 0;
}
