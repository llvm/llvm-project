// RUN: %check_clang_tidy -std=c++20-or-later %s readability-container-contains %t

// Some *very* simplified versions of `map` etc.
namespace std {

template <class Key, class T>
struct map {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const;
  void *find(const Key &K);
  void *end();
};

template <class Key>
struct set {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const;
};

template <class Key>
struct unordered_set {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const;
};

template <class Key, class T>
struct multimap {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const;
};

} // namespace std

// Check that we detect various common ways to check for membership
int testDifferentCheckTypes(std::map<int, int> &MyMap) {
  if (MyMap.count(0))
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use 'contains' to check for membership [readability-container-contains]
    // CHECK-FIXES: if (MyMap.contains(0))
    return 1;
  bool C1 = MyMap.count(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C1 = MyMap.contains(1);
  auto C2 = static_cast<bool>(MyMap.count(1));
  // CHECK-MESSAGES: :[[@LINE-1]]:37: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C2 = static_cast<bool>(MyMap.contains(1));
  auto C3 = MyMap.count(2) != 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C3 = MyMap.contains(2);
  auto C4 = MyMap.count(3) > 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C4 = MyMap.contains(3);
  auto C5 = MyMap.count(4) >= 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C5 = MyMap.contains(4);
  auto C6 = MyMap.find(5) != MyMap.end();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C6 = MyMap.contains(5);
  return C1 + C2 + C3 + C4 + C5 + C6;
}

// Check that we detect various common ways to check for non-membership
int testNegativeChecks(std::map<int, int> &MyMap) {
  bool C1 = !MyMap.count(-1);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C1 = !MyMap.contains(-1);
  auto C2 = MyMap.count(-2) == 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C2 = !MyMap.contains(-2);
  auto C3 = MyMap.count(-3) <= 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C3 = !MyMap.contains(-3);
  auto C4 = MyMap.count(-4) < 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C4 = !MyMap.contains(-4);
  auto C5 = MyMap.find(-5) == MyMap.end();
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C5 = !MyMap.contains(-5);
  return C1 + C2 + C3 + C4 + C5;
}

// Check for various types
int testDifferentTypes(std::map<int, int> &M, std::unordered_set<int> &US, std::set<int> &S, std::multimap<int, int> &MM) {
  bool C1 = M.count(1001);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C1 = M.contains(1001);
  bool C2 = US.count(1002);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C2 = US.contains(1002);
  bool C3 = S.count(1003);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C3 = S.contains(1003);
  bool C4 = MM.count(1004);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C4 = MM.contains(1004);
  return C1 + C2 + C3 + C4;
}

// The check detects all kind of `const`, reference, rvalue-reference and value types.
int testQualifiedTypes(std::map<int, int> ValueM, std::map<int, int> &RefM, const std::map<int, int> &ConstRefM, std::map<int, int> &&RValueM) {
  bool C1 = ValueM.count(2001);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C1 = ValueM.contains(2001);
  bool C2 = RefM.count(2002);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C2 = RefM.contains(2002);
  bool C3 = ConstRefM.count(2003);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C3 = ConstRefM.contains(2003);
  bool C4 = RValueM.count(2004);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C4 = RValueM.contains(2004);
  return C1 + C2 + C3 + C4;
}

// This is effectively a membership check, as the result is implicitly casted
// to `bool`.
bool returnContains(std::map<int, int> &M) {
  return M.count(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: return M.contains(42);
}

// This returns the actual count and should not be rewritten
int actualCount(std::multimap<int, int> &M) {
  return M.count(21);
  // NO-WARNING.
  // CHECK-FIXES: return M.count(21);
}

// Check that we are not confused by aliases
namespace s2 = std;
using MyMapT = s2::map<int, int>;
int typeAliases(MyMapT &MyMap) {
  bool C1 = MyMap.count(99);
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: bool C1 = MyMap.contains(99);
  return C1;
}

// Check that the tests also trigger for a local variable and not only for
// function arguments.
bool localVar() {
  using namespace std;
  map<int, int> LocalM;
  return LocalM.count(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: return LocalM.contains(42);
}

// Check various usages of an actual `count` which isn't rewritten
int nonRewrittenCount(std::multimap<int, int> &MyMap) {
  // This is an actual test if we have at least 2 usages. Shouldn't be rewritten.
  bool C1 = MyMap.count(1) >= 2;
  // NO-WARNING.
  // CHECK-FIXES: bool C1 = MyMap.count(1) >= 2;

  // "< 0" makes little sense and is always `false`. Still, let's ensure we
  // don't accidentally rewrite it to 'contains'.
  bool C2 = MyMap.count(2) < 0;
  // NO-WARNING.
  // CHECK-FIXES: bool C2 = MyMap.count(2) < 0;

  // The `count` is used in some more complicated formula.
  bool C3 = MyMap.count(1) + MyMap.count(2) * 2 + MyMap.count(3) / 3 >= 20;
  // NO-WARNING.
  // CHECK-FIXES: bool C3 = MyMap.count(1) + MyMap.count(2) * 2 + MyMap.count(3) / 3 >= 20;

  // This could theoretically be rewritten into a 'contains' after removig the
  // `4` on both sides of the comparison. For the time being, we don't detect
  // this case.
  bool C4 = MyMap.count(1) + 4 > 4;
  // NO-WARNING.
  // CHECK-FIXES: bool C4 = MyMap.count(1) + 4 > 4;

  return C1 + C2 + C3 + C4;
}

// Check different integer literal suffixes
int testDifferentIntegerLiteralSuffixes(std::map<int, int> &MyMap) {

  auto C1 = MyMap.count(2) != 0U;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C1 = MyMap.contains(2);
  auto C2 = MyMap.count(2) != 0UL;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C2 = MyMap.contains(2);
  auto C3 = 0U != MyMap.count(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C3 = MyMap.contains(2);
  auto C4 = 0UL != MyMap.count(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C4 = MyMap.contains(2);
  auto C5 = MyMap.count(2) < 1U;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C5 = !MyMap.contains(2);
  auto C6 = MyMap.count(2) < 1UL;
  // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C6 = !MyMap.contains(2);
  auto C7 = 1U > MyMap.count(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C7 = !MyMap.contains(2);
  auto C8 = 1UL > MyMap.count(2);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: auto C8 = !MyMap.contains(2);

  return C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8;
}

// We don't want to rewrite if the `contains` call is from a macro expansion
int testMacroExpansion(std::unordered_set<int> &MySet) {
#define COUNT_ONES(SET) SET.count(1)
  // Rewriting the macro would break the code
  // CHECK-FIXES: #define COUNT_ONES(SET) SET.count(1)
  // We still want to warn the user even if we don't offer a fixit
  if (COUNT_ONES(MySet)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use 'contains' to check for membership [readability-container-contains]
    // CHECK-MESSAGES: note: expanded from macro 'COUNT_ONES'
    return COUNT_ONES(MySet);
  }
#undef COUNT_ONES
#define COUNT_ONES count(1)
  // Rewriting the macro would break the code
  // CHECK-FIXES: #define COUNT_ONES count(1)
  // We still want to warn the user even if we don't offer a fixit
  if (MySet.COUNT_ONES) {
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use 'contains' to check for membership [readability-container-contains]
    // CHECK-MESSAGES: note: expanded from macro 'COUNT_ONES'
    return MySet.COUNT_ONES;
  }
#undef COUNT_ONES
#define MY_SET MySet
  // CHECK-FIXES: #define MY_SET MySet
  // We still want to rewrite one of the two calls to `count`
  if (MY_SET.count(1)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: use 'contains' to check for membership [readability-container-contains]
    // CHECK-FIXES: if (MY_SET.contains(1)) {
    return MY_SET.count(1);
  }
#undef MY_SET
  return 0;
}

// The following map has the same interface as `std::map`.
template <class Key, class T>
struct CustomMap {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const;
  void *find(const Key &K);
  void *end();
};

void testDifferentCheckTypes(CustomMap<int, int> &MyMap) {
  if (MyMap.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap.contains(0)) {};

  MyMap.find(0) != MyMap.end();
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: MyMap.contains(0);
}

struct MySubmap : public CustomMap<int, int> {};

void testSubclass(MySubmap& MyMap) {
  if (MyMap.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap.contains(0)) {};
}

using UsingMap = CustomMap<int, int>;
struct MySubmap2 : public UsingMap {};
using UsingMap2 = MySubmap2;

void testUsing(UsingMap2& MyMap) {
  if (MyMap.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap.contains(0)) {};
}

template <class Key, class T>
struct CustomMapContainsDeleted {
  unsigned count(const Key &K) const;
  bool contains(const Key &K) const = delete;
  void *find(const Key &K);
  void *end();
};

struct SubmapContainsDeleted : public CustomMapContainsDeleted<int, int> {};

void testContainsDeleted(CustomMapContainsDeleted<int, int> &MyMap,
                         SubmapContainsDeleted &MyMap2) {
  // No warning if the `contains` method is deleted.
  if (MyMap.count(0)) {};
  if (MyMap2.count(0)) {};
}

template <class Key, class T>
struct CustomMapPrivateContains {
  unsigned count(const Key &K) const;
  void *find(const Key &K);
  void *end();

private:
  bool contains(const Key &K) const;
};

struct SubmapPrivateContains : public CustomMapPrivateContains<int, int> {};

void testPrivateContains(CustomMapPrivateContains<int, int> &MyMap,
                         SubmapPrivateContains &MyMap2) {
  // No warning if the `contains` method is not public.
  if (MyMap.count(0)) {};
  if (MyMap2.count(0)) {};
}

struct MyString {};

struct WeirdNonMatchingContains {
  unsigned count(char) const;
  bool contains(const MyString&) const;
};

void testWeirdNonMatchingContains(WeirdNonMatchingContains &MyMap) {
  // No warning if there is no `contains` method with the right type.
  if (MyMap.count('a')) {};
}

template <class T>
struct SmallPtrSet {
  using ConstPtrType = const T*;
  unsigned count(ConstPtrType Ptr) const;
  bool contains(ConstPtrType Ptr) const;
};

template <class T>
struct SmallPtrPtrSet {
  using ConstPtrType = const T**;
  unsigned count(ConstPtrType Ptr) const;
  bool contains(ConstPtrType Ptr) const;
};

template <class T>
struct SmallPtrPtrPtrSet {
  using ConstPtrType = const T***;
  unsigned count(ConstPtrType Ptr) const;
  bool contains(ConstPtrType Ptr) const;
};

void testSmallPtrSet(const int ***Ptr,
                     SmallPtrSet<int> &MySet,
                     SmallPtrPtrSet<int> &MySet2,
                     SmallPtrPtrPtrSet<int> &MySet3) {
  if (MySet.count(**Ptr)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MySet.contains(**Ptr)) {};
  if (MySet2.count(*Ptr)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MySet2.contains(*Ptr)) {};
  if (MySet3.count(Ptr)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MySet3.contains(Ptr)) {};
}

struct X {};
struct Y : public X {};

void testSubclassEntry(SmallPtrSet<X>& Set, Y* Entry) {
  if (Set.count(Entry)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Set.contains(Entry)) {}
}

struct WeirdPointerApi {
  unsigned count(int** Ptr) const;
  bool contains(int* Ptr) const;
};

void testWeirdApi(WeirdPointerApi& Set, int* E) {
  if (Set.count(&E)) {}
}

void testIntUnsigned(std::set<int>& S, unsigned U) {
  if (S.count(U)) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (S.contains(U)) {}
}

template <class T>
struct CustomSetConvertible {
  unsigned count(const T &K) const;
  bool contains(const T &K) const;
};

struct A {};
struct B { B() = default; B(const A&) {} };
struct C { operator A() const; };

void testConvertibleTypes() {
  CustomSetConvertible<B> MyMap;
  if (MyMap.count(A())) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap.contains(A())) {};

  CustomSetConvertible<A> MyMap2;
  if (MyMap2.count(C())) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap2.contains(C())) {};

  if (MyMap2.count(C()) != 0) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (MyMap2.contains(C())) {};
}

template<class U>
using Box = const U& ;

template <class T>
struct CustomBoxedSet {
  unsigned count(Box<T> K) const;
  bool contains(Box<T> K) const;
};

void testBox() {
  CustomBoxedSet<int> Set;
  if (Set.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Set.contains(0)) {};
}

void testOperandPermutations(std::map<int, int>& Map) {
  if (Map.count(0) != 0) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Map.contains(0)) {};
  if (0 != Map.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Map.contains(0)) {};
  if (Map.count(0) == 0) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (!Map.contains(0)) {};
  if (0 == Map.count(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (!Map.contains(0)) {};
  if (Map.find(0) != Map.end()) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Map.contains(0)) {};
  if (Map.end() != Map.find(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (Map.contains(0)) {};
  if (Map.find(0) == Map.end()) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (!Map.contains(0)) {};
  if (Map.end() == Map.find(0)) {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: use 'contains' to check for membership [readability-container-contains]
  // CHECK-FIXES: if (!Map.contains(0)) {};
}
