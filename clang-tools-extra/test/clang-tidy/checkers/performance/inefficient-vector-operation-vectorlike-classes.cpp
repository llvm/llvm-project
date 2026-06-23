// RUN: %check_clang_tidy %s performance-inefficient-vector-operation %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  {performance-inefficient-vector-operation.VectorLikeClasses: \
// RUN:   "VectorLikeInheritedPushBack;VectorLikeDirectPushBack;VectorLikeInheritedEmplaceBack"}}'

class VectorLikePushBackBase {
public:
  void push_back(int) {}
};

class VectorLikeInheritedPushBack : public VectorLikePushBackBase {
public:
  void reserve(int);
};

class VectorLikeDirectPushBack {
public:
  void push_back(int) {}
  void reserve(int) {}
};

class VectorLikeEmplaceBackBase {
public:
  void emplace_back(int) {}
};

class VectorLikeInheritedEmplaceBack : public VectorLikeEmplaceBackBase {
public:
  void reserve(int);
};

void testVectorLikeClasses() {
  {
    VectorLikeInheritedPushBack inheritedPushBackVector;
    // CHECK-FIXES: inheritedPushBackVector.reserve(100);
    for (int I = 0; I < 100; ++I) {
      inheritedPushBackVector.push_back(I);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called inside a loop; consider pre-allocating the container capacity before the loop
    }
  }

  {
    VectorLikeDirectPushBack directPushBackVector;
    // CHECK-FIXES: directPushBackVector.reserve(100);
    for (int I = 0; I < 100; ++I) {
      directPushBackVector.push_back(I);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'push_back' is called inside a loop; consider pre-allocating the container capacity before the loop
    }
  }

  {
    VectorLikeInheritedEmplaceBack inheritedEmplaceBackVector;
    // CHECK-FIXES: inheritedEmplaceBackVector.reserve(100);
    for (int I = 0; I < 100; ++I) {
      inheritedEmplaceBackVector.emplace_back(I);
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'emplace_back' is called inside a loop; consider pre-allocating the container capacity before the loop
    }
  }
}
