// RUN: %check_clang_tidy -std=c++11 %s performance-inefficient-algorithm %t
// RUN: %check_clang_tidy -std=c++14,c++17 %s performance-inefficient-algorithm %t
// RUN: %check_clang_tidy -std=c++20-or-later %s performance-inefficient-algorithm %t

namespace std {
template <typename T> struct less {
  bool operator()(const T &lhs, const T &rhs) { return lhs < rhs; }
};

template <> struct less<void> {
  template <typename T, typename U>
  bool operator()(const T &lhs, const U &rhs) const { return lhs < rhs; }
};

template <typename T> struct greater {
  bool operator()(const T &lhs, const T &rhs) { return lhs > rhs; }
};

template <typename T> struct allocator {};

struct iterator_type {};

template <typename K, typename Cmp = less<K>, typename Alloc = allocator<K>>
struct set {
  typedef iterator_type iterator;
  iterator find(const K &k);
  unsigned count(const K &k);
  iterator lower_bound(const K &k);
  iterator upper_bound(const K &k);

  iterator begin();
  iterator end();
  iterator begin() const;
  iterator end() const;
};

struct other_iterator_type {};

template <typename K, typename V, typename Cmp = less<K>> struct map {
  typedef other_iterator_type iterator;
  iterator find(const K &k);
  unsigned count(const K &k);

  iterator begin();
  iterator end();
  iterator begin() const;
  iterator end() const;
};

template <typename K, typename V> struct multimap : map<K, V> {};
template <typename K> struct unordered_set : set<K> {};
template <typename K, typename V> struct unordered_map : map<K, V> {};
template <typename K> struct unordered_multiset : set<K> {};
template <typename K, typename V> struct unordered_multimap : map<K, V> {};

template <typename K, typename Cmp = less<K>, typename Alloc = allocator<K>>
struct multiset : set<K, Cmp, Alloc> {};

template <typename FwIt, typename K>
FwIt find(FwIt, FwIt end, const K &) { return end; }

template <typename FwIt, typename K, typename Cmp>
FwIt find(FwIt, FwIt end, const K &, Cmp) { return end; }

template <typename FwIt, typename Pred>
FwIt find_if(FwIt, FwIt end, Pred) { return end; }

template <typename FwIt, typename K>
unsigned count(FwIt, FwIt, const K &) { return 0; }

template <typename FwIt, typename K>
FwIt lower_bound(FwIt, FwIt end, const K &) { return end; }

template <typename FwIt, typename K, typename Ord>
FwIt lower_bound(FwIt, FwIt end, const K &, Ord) { return end; }
}

#define FIND_IN_SET(x) find(x.begin(), x.end(), 10)
// CHECK-FIXES: #define FIND_IN_SET(x) find(x.begin(), x.end(), 10)

template <typename T> void f(const T &t) {
  std::set<int> s;
  find(s.begin(), s.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(46);

  find(t.begin(), t.end(), 46);
  // CHECK-FIXES: find(t.begin(), t.end(), 46);
}

int main() {
  std::set<int> s;
  auto it = std::find(s.begin(), s.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: this STL algorithm call should be replaced with the container method 'find' [performance-inefficient-algorithm]
  // CHECK-FIXES: auto it = s.find(43);
  auto c = count(s.begin(), s.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: auto c = s.count(43);
  auto p = std::find(s.begin(), s.end(), (43));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: auto p = s.find((43));
  auto r = std::find((s).begin(), (s).end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: auto r = s.find(43);
  int i = 1, j = 2;
  auto q = std::find(s.begin(), s.end(), (i++, j));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: auto q = s.find((i++, j));

#define SECOND(x, y, z) y
  SECOND(q,std::count(s.begin(), s.end(), 22),w);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be
  // CHECK-FIXES: SECOND(q,s.count(22),w);

  it = find_if(s.begin(), s.end(), [](int) { return false; });

  std::multiset<int> ms;
  find(ms.begin(), ms.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: ms.find(46);

  const std::multiset<int> &msref = ms;
  find(msref.begin(), msref.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: msref.find(46);

  std::multiset<int> *msptr = &ms;
  find(msptr->begin(), msptr->end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: msptr->find(46);

  find((msptr)->begin(), (msptr)->end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: msptr->find(46);

  it = std::find(s.begin(), s.end(), 43, std::greater<int>());
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: different comparers used in the algorithm and the container [performance-inefficient-algorithm]

  FIND_IN_SET(s);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: FIND_IN_SET(s);

  f(s);

  std::unordered_set<int> us;
  lower_bound(us.begin(), us.end(), 10);
  // CHECK-FIXES: lower_bound(us.begin(), us.end(), 10);
  find(us.begin(), us.end(), 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: us.find(10);

  std::unordered_multiset<int> ums;
  find(ums.begin(), ums.end(), 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: ums.find(10);

  std::map<int, int> intmap;
  find(intmap.begin(), intmap.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(intmap.begin(), intmap.end(), 46);

  std::multimap<int, int> intmmap;
  find(intmmap.begin(), intmmap.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(intmmap.begin(), intmmap.end(), 46);

  std::unordered_map<int, int> umap;
  find(umap.begin(), umap.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(umap.begin(), umap.end(), 46);

  std::unordered_multimap<int, int> ummap;
  find(ummap.begin(), ummap.end(), 46);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(ummap.begin(), ummap.end(), 46);
}

struct Value {
  int value;
};

struct Ordering {
  bool operator()(const Value &lhs, const Value &rhs) const {
    return lhs.value < rhs.value;
  }
  bool operator()(int lhs, const Value &rhs) const { return lhs < rhs.value; }
};

void g(std::set<Value, Ordering> container, int value) {
  lower_bound(container.begin(), container.end(), value, Ordering());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: lower_bound(container.begin(), container.end(), value, Ordering());
}

#define PAREN_VALUE (43)
#define VALUE_OF(x) ((x) + 1)
#define PLAIN_VALUE 43
#define VAL_AND_END s.end(), 46

// The searched-for value is copied as written, so a whole macro expansion is
// kept intact, and so is a range that just starts or ends inside one. A value
// that covers only part of an expansion has no source text of its own, and the
// call is then diagnosed without a fix.
void macroExpansion(std::set<int> s, int i) {
  find(s.begin(), s.end(), PAREN_VALUE);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(PAREN_VALUE);

  count(s.begin(), s.end(), VALUE_OF(i));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.count(VALUE_OF(i));

  find(s.begin(), s.end(), PLAIN_VALUE);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(PLAIN_VALUE);

  find(s.begin(), s.end(), PLAIN_VALUE + i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(PLAIN_VALUE + i);

  find(s.begin(), VAL_AND_END);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(s.begin(), VAL_AND_END);
}

#define PAREN_CONT (s)
#define PLAIN_CONT s
#define RANGE s.begin(), s.end()
#define PTR_RANGE p->begin(), p->end()
#define CONT_AND_BEGIN s.begin()
#define BARE_RANGE_OF(c) c.begin(), c.end()
#define PAREN_RANGE_OF(c) (c).begin(), (c).end()
#define MY_SET s
#define NESTED_RANGE MY_SET.begin(), MY_SET.end()

// The container text comes from the reference to it, with any parentheses
// around it stripped. A reference spelled where the macro was expanded, such as
// a macro argument, is read from there, and one that spans a whole expansion
// keeps it intact. A reference covering only part of an expansion has no source
// text of its own, and the call is then diagnosed without a fix.
void macroContainer(std::set<int> s, std::set<int> *p) {
  find(BARE_RANGE_OF(s), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(43);

  find(BARE_RANGE_OF(MY_SET), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: MY_SET.find(43);

  find(PLAIN_CONT.begin(), PLAIN_CONT.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: PLAIN_CONT.find(43);

  find(RANGE, 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(RANGE, 43);

  count(PTR_RANGE, 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: count(PTR_RANGE, 43);

  find(CONT_AND_BEGIN, s.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(CONT_AND_BEGIN, s.end(), 43);

  find(NESTED_RANGE, 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(NESTED_RANGE, 43);

  find(PAREN_RANGE_OF(s), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.find(43);

  find(PAREN_CONT.begin(), PAREN_CONT.end(), 43);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(PAREN_CONT.begin(), PAREN_CONT.end(), 43);

  find(RANGE, PLAIN_VALUE);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find(RANGE, PLAIN_VALUE);
}

int getBound();

void findIf(std::set<int> s, std::multiset<int> ms, int i) {
  auto a = std::find_if(s.begin(), s.end(), [&](int val) { return val > i; });
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be replaced with the container method 'upper_bound' [performance-inefficient-algorithm]
  // CHECK-FIXES: auto a = s.upper_bound(i);

  auto b = find_if(s.begin(), s.end(), [&](int val) { return val >= i; });
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: this STL algorithm call should be replaced with the container method 'lower_bound' [performance-inefficient-algorithm]
  // CHECK-FIXES: auto b = s.lower_bound(i);

  find_if(s.begin(), s.end(), [&](int val) { return i < val; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be replaced with the container method 'upper_bound'
  // CHECK-FIXES: s.upper_bound(i);

  find_if(s.begin(), s.end(), [&](int val) { return i <= val; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be replaced with the container method 'lower_bound'
  // CHECK-FIXES: s.lower_bound(i);

  // The bound is copied as written.
  find_if(s.begin(), s.end(), [&](const int &val) { return val > (i); });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.upper_bound((i));

  find_if(ms.begin(), ms.end(), [&](int val) { return val > i; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: ms.upper_bound(i);

  // The element has to be on the side the container method searches from.
  find_if(s.begin(), s.end(), [&](int val) { return val < i; });
  find_if(s.begin(), s.end(), [&](int val) { return i > val; });

  // The bound moves to the call site, so it must not name the element or a
  // second parameter, and evaluating it once must be enough.
  find_if(s.begin(), s.end(), [&](int val) { return val > i - val; });
  find_if(s.begin(), s.end(), [&](int val, int k = 0) { return val > k; });
  find_if(s.begin(), s.end(), [&](int val) { return val > getBound(); });
  volatile int vi = 0;
  find_if(s.begin(), s.end(), [&](int val) { return val > vi; });

  find_if(s.begin(), s.end(), [&](int val) {
    int j = i;
    return val > j;
  });

  std::set<int> other;
  find_if(s.begin(), other.end(), [&](int val) { return val > i; });
}

void findIfTransparentComparator(std::set<int, std::less<void>> s, int i) {
  find_if(s.begin(), s.end(), [&](int val) { return val > i; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: s.upper_bound(i);
}

struct Bounded {
  int Threshold;
  void findIf(std::set<int> s) {
    find_if(s.begin(), s.end(), [this](int val) { return val > Threshold; });
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: this STL algorithm call should be
    // CHECK-FIXES: s.upper_bound(Threshold);
  }
};

// A container that does not binary search by `<`, or a predicate that is not a
// comparison written as a lambda right at the call.
void findIfUnsupported(std::unordered_set<int> us,
                       std::set<int, std::greater<int>> desc, std::set<int> s,
                       int i) {
  find_if(us.begin(), us.end(), [&](int val) { return val > i; });
  find_if(desc.begin(), desc.end(), [&](int val) { return val > i; });
  auto pred = [&](int val) { return val > i; };
  find_if(s.begin(), s.end(), pred);
  find_if(s.begin(), s.end(), [&](int val) { return val == i; });
}

enum Color { Red, Green };

// `<` and `>` have to be the built-in operators which order the elements the
// same way `std::less` does.
void findIfKeyType(std::set<int *> sp, std::set<Color> sc, std::set<double> sd,
                   int *p, Color c, double d) {
  find_if(sp.begin(), sp.end(), [&](int *val) { return val > p; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: sp.upper_bound(p);
  find_if(sc.begin(), sc.end(), [&](Color val) { return val > c; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: sc.upper_bound(c);

  find_if(sd.begin(), sd.end(), [&](double val) { return val >= d; });
}

// The bound may differ in type from the key as long as converting it to the key
// type cannot change its value.
void findIfConvertedBound(std::set<long long> sll, std::set<int> si,
                          std::set<unsigned char> suc, std::set<unsigned> su,
                          std::set<Color> sc, int i, long long ll,
                          unsigned short ush) {
  find_if(sll.begin(), sll.end(), [&](long long val) { return val > i; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: sll.upper_bound(i);
  find_if(si.begin(), si.end(), [&](int val) { return val > ush; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: si.upper_bound(ush);
  find_if(suc.begin(), suc.end(), [](unsigned char val) { return val > 100; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: suc.upper_bound(100);
  find_if(su.begin(), su.end(), [](unsigned val) { return val > 43u; });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: su.upper_bound(43u);

  find_if(si.begin(), si.end(), [&](int val) { return val > ll; });
  find_if(suc.begin(), suc.end(), [](unsigned char val) { return val > -1; });
  find_if(sc.begin(), sc.end(), [](Color val) { return val > 1; });
  find_if(si.begin(), si.end(), [](int val) { return val > 43u; });
  find_if(sll.begin(), sll.end(), [&](int val) { return val > i; });
}

// A lambda in a template has a dependent closure type, so the pattern holds an
// unresolved call and only the instantiation is checked, where the parameter
// type is `short` and differs from the key.
template <typename T>
void findIfTemplate(std::set<int> s, int i) {
  find_if(s.begin(), s.end(), [&](T val) { return val > i; });
}

void useFindIfTemplate(std::set<int> s, int i) { findIfTemplate<short>(s, i); }

#define GT_BOUND(v) v > 43
#define FIND_IF_GT(c, b) find_if(c.begin(), c.end(), [&](int val) { return val > b; })
// CHECK-FIXES: #define FIND_IF_GT(c, b) find_if(c.begin(), c.end(), [&](int val) { return val > b; })

// Neither a bound covering only part of a macro expansion nor a call spelled
// inside one has source text of its own.
void macroBound(std::set<int> s, int i) {
  find_if(s.begin(), s.end(), [](int val) { return GT_BOUND(val); });
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be
  // CHECK-FIXES: find_if(s.begin(), s.end(), [](int val) { return GT_BOUND(val); });

  FIND_IF_GT(s, i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: this STL algorithm call should be replaced with the container method 'upper_bound'
  // CHECK-FIXES: FIND_IF_GT(s, i);
}

#if __cplusplus >= 201402L
// An init-capture is not in scope at the call site, and dropping the lambda
// would drop its initializer. A generic lambda's parameter is not the key type.
void findIfCxx14(std::set<long> s, long b) {
  find_if(s.begin(), s.end(), [bound = b](long val) { return val > bound; });
  find_if(s.begin(), s.end(), [c = getBound()](long val) { return val > 43; });
  find_if(s.begin(), s.end(), [&](auto val) { return val > b; });
}
#endif

enum Ordered { Low, High };
bool operator>(Ordered, Ordered);

void findIfOverloadedComparison(std::set<Ordered> s, Ordered k) {
  find_if(s.begin(), s.end(), [&](Ordered val) { return val > k; });
}

#if __cplusplus >= 201402L
// A generic lambda stays rejected whether or not an unrelated `operator>` is in
// scope to spell its dependent body as an overloaded call.
void findIfGenericLambdaWithVisibleOperator(std::set<int> s, int i) {
  find_if(s.begin(), s.end(), [&](auto val) { return val > i; });
}
#endif

#if __cplusplus >= 202002L
struct ThreeWay {
  bool operator>(int) const;
};

enum Rewritten { First, Second };
ThreeWay operator<=>(Rewritten, Rewritten);

void findIfRewrittenComparison(std::set<Rewritten> s, Rewritten k) {
  find_if(s.begin(), s.end(), [&](Rewritten val) { return val > k; });
}

// An explicit template parameter list makes the lambda generic too.
void findIfLambdaTemplateParameter(std::set<int> s) {
  find_if(s.begin(), s.end(), [&]<int N = 5>(int val) { return val > N; });
}
#endif
