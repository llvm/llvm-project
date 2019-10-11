// RUN: %check_clang_tidy %s cppcoreguidelines-owning-memory %t

namespace gsl {
template <typename T>
using owner = T;
}

namespace std {

// Not actually a vector, but more a dynamic, fixed size array. Just to demonstrate
// functionality or the lack of the same.
template <typename T>
class vector {
public:
  vector(unsigned long size, T val) : data{new T[size]}, size{size} {
    for (unsigned long i = 0ul; i < size; ++i) {
      data[i] = val;
    }
  }

  T *begin() { return data; }
  T *end() { return &data[size]; }
  T &operator[](unsigned long index) { return data[index]; }

private:
  T *data;
  unsigned long size;
};

} // namespace std

// All of the following codesnippets should be valid with appropriate 'owner<>' anaylsis,
// but currently the type information of 'gsl::owner<>' gets lost in typededuction.
int main() {
  std::vector<gsl::owner<int *>> OwnerStdVector(100, nullptr);

  // Rangebased looping in resource vector.
  for (auto *Element : OwnerStdVector) {
    Element = new int(42);
    // CHECK-NOTES: [[@LINE-1]]:5: warning: assigning newly created 'gsl::owner<>' to non-owner 'int *'
  }
  for (auto *Element : OwnerStdVector) {
    delete Element;
    // CHECK-NOTES: [[@LINE-1]]:5: warning: deleting a pointer through a type that is not marked 'gsl::owner<>'; consider using a smart pointer instead
    // CHECK-NOTES: [[@LINE-3]]:8: note: variable declared here
  }

  // Indexbased looping in resource vector.
  for (int i = 0; i < 100; ++i) {
    OwnerStdVector[i] = new int(42);
    // CHECK-NOTES: [[@LINE-1]]:5: warning: assigning newly created 'gsl::owner<>' to non-owner 'int *'
  }
  for (int i = 0; i < 100; ++i) {
    delete OwnerStdVector[i];
    // CHECK-NOTES: [[@LINE-1]]:5: warning: deleting a pointer through a type that is not marked 'gsl::owner<>'; consider using a smart pointer instead
    // CHECK-NOTES: [[@LINE-21]]:3: note: variable declared here
    // A note gets emitted here pointing to the return value of the operator[] from the
    // vector implementation. Maybe this is considered misleading.
  }

  return 0;
}
