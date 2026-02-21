// RUN: %check_clang_tidy -check-suffix=,CPP20ORLATER -std=c++20-or-later %s modernize-use-structured-binding %t -- -- -I %S/Inputs/use-structured-binding/
// RUN: %check_clang_tidy -std=c++17 %s modernize-use-structured-binding %t -- -- -I %S/Inputs/use-structured-binding/
#include "fake_std_pair_tuple.h"

template<typename T>
void MarkUsed(T x);

struct TestClass {
  int a;
  int b;
  TestClass() : a(0), b(0) {}
  TestClass& operator++();
  TestClass(int x, int y) : a(x), b(y) {}
};

void DecomposeByAssignWarnCases() {
  {
    auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    int x = P.first, y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    int x = P.first, y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
    int z;
  }

  {
    auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    auto y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    const auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto [x, y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    const int x = P.first;
    const auto y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    std::pair<int, int> otherP;
    auto& P = otherP;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto& [x, y] = otherP;
    // CHECK-NEXT: // REMOVE
    int& x = P.first;
    auto& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    std::pair<int, int> otherP;
    const auto& P = otherP;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto& [x, y] = otherP;
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    const auto& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    auto P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
    
    auto another_p = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [another_x, another_y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    int another_x = another_p.first;
    int another_y = another_p.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }
}

void forRangeWarnCases() {
  std::pair<int, int> Pairs[10];
  for (auto P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto [x, y] : Pairs) {
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  for (auto P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto [x, y] : Pairs) {
    int x = P.first, y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  for (auto P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto [x, y] : Pairs) {
    int x = P.first, y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
    int z;
  }

  for (const auto P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (const auto [x, y] : Pairs) {
    // CHECK-NEXT: // REMOVE
    const int x = P.first;
    const int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  for (auto& P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto& [x, y] : Pairs) {
    // CHECK-NEXT: // REMOVE
    int& x = P.first;
    int& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  for (const auto& P : Pairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (const auto& [x, y] : Pairs) {
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    const int& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  std::pair<TestClass, TestClass> ClassPairs[10];
  for (auto P : ClassPairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto [c1, c2] : ClassPairs) {
    // CHECK-NEXT: // REMOVE
    TestClass c1 = P.first;
    TestClass c2 = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  for (const auto P : ClassPairs) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (const auto [c1, c2] : ClassPairs) {
    // CHECK-NEXT: // REMOVE
    const TestClass c1 = P.first;
    const TestClass c2 = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }
}

void forRangeNotWarnCases() {
  std::pair<int, int> Pairs[10];
  for (auto P : Pairs) {
    int x = P.first;
    MarkUsed(x);
    int y = P.second;
  }

  for (auto P : Pairs) {
    MarkUsed(P);
    int x = P.first;
    int y = P.second;
  }

  for (auto P : Pairs) {
    int x = P.first;
    int y = P.second;
    MarkUsed(P);
  }

  std::pair<TestClass, TestClass> ClassPairs[10];
  for (auto P : ClassPairs) {
    TestClass c1 = P.first;
    ++ c1 ;
    TestClass c2 = P.second;
  }

  int c;
  for (auto P : ClassPairs) {
    TestClass c1 = P.first;
    c ++ ;
    TestClass c2 = P.second;
  }
}

void stdTieWarnCases() {
  // CHECK-NEXT: // REMOVE
  int a = 0;
  int b = 0; // REMOVE
  // CHECK-FIXES: // REMOVE
  std::tie(a, b) = getPair<int, int>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES: auto [a, b] = getPair<int, int>();

  int x = 0, y = 0; // REMOVE
  // CHECK-FIXES: // REMOVE
  std::tie(x, y) = getPair<int, int>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES: auto [x, y] = getPair<int, int>();

  // CHECK-NEXT: // REMOVE
  int* pa = nullptr;
  int* pb = nullptr; // REMOVE
  // CHECK-FIXES: // REMOVE
  std::tie(pa, pb) = getPair<int*, int*>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES: auto [pa, pb] = getPair<int*, int*>();

  // CHECK-NEXT: // REMOVE
  TestClass c1 (1, 2);
  TestClass c2 = TestClass {3, 4}; // REMOVE
  // CHECK-FIXES: // REMOVE
  std::tie(c1, c2) = getPair<TestClass, TestClass>();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES: auto [c1, c2] = getPair<TestClass, TestClass>();
}

void stdTieNotWarnCases() {
  int a = 0;
  int b = 0;
  a = 4;
  std::tie(a, b) = getPair<int, int>(); // no warning

  int c = 0, d = 0;
  int e = 0;
  std::tie(a, b) = getPair<int, int>(); // no warning

  int* pa = nullptr;
  int* pb = nullptr;
  MarkUsed(pa);
  std::tie(pa, pb) = getPair<int*, int*>(); // no warning

  TestClass c1 (1, 2);
  TestClass c2 = TestClass {3, 4};
  MarkUsed(c2);
  std::tie(c1, c2) = getPair<TestClass, TestClass>();
}

void NotWarnForVarHasSpecifiers() {
  {
    auto P = getPair<int, int>();
    const int x = P.first;
    int y = P.second;
  }

  {
    auto P = getPair<int, int>();
    volatile int x = P.first;
    int y = P.second;
  }

  {
    auto P = getPair<int, int>();
    int x = P.first;
    [[maybe_unused]] int y = P.second;
  }

  {
    static auto P = getPair<int, int>();
    int x = P.first;
    int y = P.second;
  }

  {
    thread_local auto P = getPair<int, int>();
    int x = P.first;
    int y = P.second;
  }

  {
    extern int a;
    int b = 0;
    std::tie(a, b) = getPair<int, int>();
  }
}

void NotWarnForMultiUsedPairVar() {
  {
    auto P = getPair<int, int>();
    int x = P.first;
    int y = P.second;
    MarkUsed(P);
  }

  {
    auto P = getPair<int, int>();
    int x = P.first;
    MarkUsed(P);
    int y = P.second;
  }

  {
    auto P = getPair<int, int>();
    MarkUsed(P);
    int x = P.first;
    int y = P.second;
  }

  {
    std::pair<int, int> Pairs[10];
    for (auto P : Pairs) {
      int x = P.first;
      int y = P.second;

      MarkUsed(P);
    }
  }
}

#define DECOMPOSE(P)                                                    \
    int x = P.first;                                                    \
    int y = P.second;                                                   \

void NotWarnForMacro1() {
  auto P = getPair<int, int>();
  DECOMPOSE(P);
}

#define GETPAIR auto P = getPair<int, int>()

void NotWarnForMacro2() {
  GETPAIR;
  int x = P.first;
  int y = P.second;
}

void captureByVal() {
  auto P = getPair<int, int>();
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: auto [x, y] = getPair<int, int>();
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
  int x = P.first;
  int y = P.second; // REMOVE
  // CHECK-FIXES-CPP20ORLATER: // REMOVE

  auto lambda = [x]() {
    int y = x;
  };
}

void captureByRef() {
  auto P = getPair<int, int>();
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: auto [x, y] = getPair<int, int>();
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
  int x = P.first;
  int y = P.second; // REMOVE
  // CHECK-FIXES-CPP20ORLATER: // REMOVE

  auto lambda = [&x]() {
    x = 1;
  };
}

void captureByAllRef() {
  auto P = getPair<int, int>();
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: auto [x, y] = getPair<int, int>();
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
  int x = P.first;
  int y = P.second; // REMOVE
  // CHECK-FIXES-CPP20ORLATER: // REMOVE

  auto lambda = [&]() {
    x = 1;
  };
}

void deepLambda() {
  auto P = getPair<int, int>();
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: auto [x, y] = getPair<int, int>();
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
  int x = P.first;
  int y = P.second; // REMOVE
  // CHECK-FIXES-CPP20ORLATER: // REMOVE

  {
    auto lambda = [x]() {
      int y = x;
    };
  }
}

void forRangeNotWarn() {
  std::pair<int, int> Pairs[10];
  for (auto P : Pairs) {
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: for (auto [x, y] : Pairs) {
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES-CPP20ORLATER: // REMOVE

    auto lambda = [&]() {
    x = 1;
  };
  }
}

void stdTieNotWarn() {
  // CHECK-NEXT-CPP20ORLATER: // REMOVE
  int x = 0;
  int y = 0; // REMOVE
  // CHECK-FIXES-CPP20ORLATER: // REMOVE
  std::tie(x, y) = getPair<int, int>();
  // CHECK-MESSAGES-CPP20ORLATER: :[[@LINE-1]]:3: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
  // CHECK-FIXES-CPP20ORLATER: auto [x, y] = getPair<int, int>();

  auto lambda = [&x]() {
    x = 1;
  };
}

struct otherPair {
  int first;
  int second;
};

void OtherPairTest() {
  {
    auto P = otherPair();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = otherPair();
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    const auto P = otherPair();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto [x, y] = otherPair();
    // CHECK-NEXT: // REMOVE
    const int x = P.first;
    const auto y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    otherPair otherP;
    auto& P = otherP;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto& [x, y] = otherP;
    // CHECK-NEXT: // REMOVE
    int& x = P.first;
    auto& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    std::pair<int, int> otherP;
    const auto& P = otherP;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto& [x, y] = otherP;
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    const auto& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }
}

void OtherPairNotWarnCases() {
  {
    auto P = otherPair();
    const int x = P.first;
    int y = P.second;
  }

  {
    auto P = otherPair();
    volatile int x = P.first;
    int y = P.second;
  }

  {
    auto P = otherPair();
    int x = P.first;
    [[maybe_unused]] int y = P.second;
  }

  {
    static auto P = getPair<int, int>();
    int x = P.first;
    int y = P.second;
  }
}

struct otherNonPair1 {
  int first;
  int second;

private:
  int third;
};

struct otherNonPair2 {
  int first;
  int second;
  int third;
};

void OtherNonPairTest() {
  {
    auto P = otherNonPair1();
    int x = P.first;
    int y = P.second;
  }

  {
    auto P = otherNonPair2();
    int x = P.first;
    int y = P.second;
  }
}

template<typename PairType>
PairType getCertainPair();

struct ConstFieldPair {
  const int first;
  int second;
};

void ConstFieldPairTests() {
  {
    const ConstFieldPair P = getCertainPair<ConstFieldPair>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto [x, y] = getCertainPair<ConstFieldPair>();
    // CHECK-NEXT: // REMOVE
    const int x = P.first;
    const int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    const ConstFieldPair& P = getCertainPair<ConstFieldPair>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto& [x, y] = getCertainPair<ConstFieldPair>();
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    const int& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    ConstFieldPair P = getCertainPair<ConstFieldPair>(); // no warning
    int x = P.first;
    int y = P.second;
  }
}

struct PointerFieldPair {
  int* first;
  int second;
};

void PointerFieldPairTests() {
  {
    PointerFieldPair P = getCertainPair<PointerFieldPair>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getCertainPair<PointerFieldPair>();
    // CHECK-NEXT: // REMOVE
    int* x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    PointerFieldPair P = getCertainPair<PointerFieldPair>(); // no warning
    const int* x = P.first;
    int y = P.second;
  }
}

struct ConstRefFieldPair {
  const int& first;
  int second;
  ConstRefFieldPair(int& f, int s) : first(f), second(s) {}
};

void ConstRefFieldPairTests() {
  {
    ConstRefFieldPair P = getCertainPair<ConstRefFieldPair>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getCertainPair<ConstRefFieldPair>();
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  {
    ConstRefFieldPair P = getCertainPair<ConstRefFieldPair>();; // no warning
    int x = P.first;
    int y = P.second;
  }
}

struct StaticFieldPair {
  static int first;
  int second;
};

void StaticFieldPairTests() {
  {
    StaticFieldPair P; // Should not warn
    int x = P.first;
    int y = P.second;
  }

  {
    StaticFieldPair P; // Should not warn
    static int x = P.first;
    int y = P.second;
  }
}

void IgnoreDirectInit() {
  {
    std::pair<int, int> P{1, 1};
    int x = P.first;
    int y = P.second; 
  }

  {
    std::pair<int, int> P(1, 1);
    int x = P.first;
    int y = P.second;
  }

  {
    std::pair<int, int> P;
    int x = P.first;
    int y = P.second;
  }
}

void StdMapTestCases() {
  for (auto p : std::unordered_map<int, int>()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: for (auto [x, y] : std::unordered_map<int, int>()) {
    // CHECK-NEXT: // REMOVE
    int x = p.first;
    int y = p.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }
}

void SpecialCases1() {
  {
    auto P = getPair<int, int>();                                                                                                                                                          
    int y = P.second;  // second first                                                                                                                                                  
    int x = P.first;
  }

  {
    auto P = getPair<int, int>();                                                                                                                                                          
    int x = P.first;                                                                                                                                                                       
    // P.second never assigned
  }

  {
    auto P = getPair<int, int>();                                                                                                                                                          
    int x = P.first + 1;                                                                                                                                             
    int y = P.second;  
  }                                                                                                                                                                                                                                                                                                                                                                                                 
  {
    auto P = getPair<int, int>();                                                                                                                                                          
    float x = P.first;  // int -> float                                                                                                                                                      
    float y = P.second;   
  }                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                           
  {
    auto P = getPair<int, int>();                                                                                                                                                          
    int x = P.first;                                                                                                                                                                       
    int dummy = 42;                                                                                                                                         
    int y = P.second;     
  }
  
  {
    for (auto P : std::unordered_map<int, int>()) {
      int x = P.first;                                                                                                                                                                     
      int dummy = 42;                                                                                                                                              
      int y = P.second;     
    }
  }
                                                                                                                                                                                          
  {
    std::pair<int, int> somePair;
    std::pair<int, int>* P = &somePair;                                                                                                                                                    
    int x = P->first;                                                                                                                                                                      
    int y = P->second;
  }

  {
    std::pair<int, int> somePair;
    std::pair<int, int>& P = somePair;                                        
    int x = P.first;                                                                                                                                                                      
    int y = P.second;
  }

  {
    std::pair<int, int> somePair;
    const std::pair<int, int>& P = somePair;                                        
    int x = P.first;                                                                                                                                                                      
    int y = P.second;
  }
}

// Partial and full templates
// Currently not supported because we can't get type from CXXDependentScopeMemberExpr, needs some extra work to do that.
template<typename T>
void templateFuncTest() {
  auto P = getPair<T, int>();
  T x = P.first;
  int y = P.second;
}

template<typename T1, typename T2>
void templateFuncTest() {
  auto P = getPair<T1, T2>();
  T1 x = P.first;
  T2 y = P.second;
}

void SpecialCases2() {
  // nested pairs
  {
    auto P = getPair<std::pair<int, int>, int>();
    int val = P.second;
    std::pair<int, int> inner = P.first;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [a_in, b_in] = P.first;
    // CHECK-NEXT: // REMOVE
    int a_in = inner.first;
    int b_in = inner.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  // shadow
  {
    int x = 10;
    {
      auto P = getPair<int, int>();
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
      // CHECK-FIXES: auto [x, y] = getPair<int, int>();
      // CHECK-NEXT: // REMOVE
      int x = P.first;
      int y = P.second; // REMOVE
      // CHECK-FIXES: // REMOVE
    }
  }

  // only use first
  {
    auto P = getPair<int, int>();
    int x = P.first;
    int y = P.first;
  }

  // std::ignore in std::tie
  {
    int a = 0;
    std::tie(a, std::ignore) = getPair<int, int>();
  }

  // tuple pair mixup
  {
    int a = 0, b = 0, c = 0;
    std::tie(a, b, c) = getTuple<int, int, int>();
    std::tie(a, a) = getPair<int, int>();
  }

  // volatile specifier
  {
    volatile std::pair<int, int> P = getPair<int, int>();
    int x = P.first;
    int y = P.second;
  }

  // decltype usage (there are invisible DeclRefExpr in decltype)
  {
    auto P = getPair<int, int>();
    decltype(P.first) x = P.first;
    decltype(P.second) y = P.second;
  }

  // constexpr pair
  {
    constexpr auto P = getConstexprPair<int, int>();
    constexpr int x = P.first;
    constexpr int y = P.second;
  }

  // initializer list
  {
    struct AggregatePair {
      int first;
      int second;
    };
    auto P = AggregatePair{.first = 1, .second = 2};
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = AggregatePair{.first = 1, .second = 2};
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  // reference to original pair
  {
    std::pair<int, int> original = getPair<int, int>();
    const auto& P = original;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: const auto& [x, y] = original;
    // CHECK-NEXT: // REMOVE
    const int& x = P.first;
    const int& y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  // typedef pair
  {
    using IntPair = std::pair<int, int>;
    IntPair P = getPair<int, int>();
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = getPair<int, int>();
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }

  // assign to existing vars
  {
    int x, y;
    auto P = getPair<int, int>();
    x = P.first;  
    y = P.second;
  }

  // conditional operator
  {
    bool cond;
    std::pair<int, int> p1{1, 2}, p2{3, 4};
    auto P = cond ? p1 : p2;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: use a structured binding to decompose a pair [modernize-use-structured-binding]
    // CHECK-FIXES: auto [x, y] = cond ? p1 : p2;
    // CHECK-NEXT: // REMOVE
    int x = P.first;
    int y = P.second; // REMOVE
    // CHECK-FIXES: // REMOVE
  }
}
