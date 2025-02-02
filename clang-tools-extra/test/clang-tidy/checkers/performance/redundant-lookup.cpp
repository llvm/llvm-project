// RUN: %check_clang_tidy %s performance-redundant-lookup %t -- -- -isystem %clang_tidy_headers

#include <map>
#include <set>

#define myassert(cond) do { if (!(cond)) myabort(); } while (0)
void myabort() __attribute__((noreturn));
int rng(int seed);
template <class T, class... Ts> void escape(T&, Ts&...);

namespace my {
template <class T, class U> class FancyMap {
public:
  using key_type = T;
  bool contains(key_type) const;
  int count(key_type) const;
  U &operator[](key_type);
  const U &operator[](key_type) const;

private:
  key_type* keys;
  U* values;
};

template <class T> class FancySet {
public:
  using key_type = T;
  bool contains(key_type) const;
  int count(key_type) const;

private:
  key_type* keys;
};
} // namespace my


void containerNameSpellsSet(my::FancySet<int> &s, int key) {
  (void)s.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups [performance-redundant-lookup]
  (void)s.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void containerNameSpellsMap(my::FancyMap<int, int> &s, int key) {
  (void)s.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups [performance-redundant-lookup]
  (void)s.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void stdSetAlsoWorks(std::set<int> &s, int key) {
  (void)s.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups [performance-redundant-lookup]
  (void)s.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void stdMapAlsoWorks(std::map<int, int> &m, int key) {
  (void)m.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups [performance-redundant-lookup]
  (void)m.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void differentLookupKeys(std::map<int, int> &m, int key, int key2, int val) {
  (void)m.contains(key);
  m[key2] = val; // no-warning: The second lookup uses a different key.
}

void differentContainers(std::map<int, int> &first, std::map<int, int> &second, int key, int val) {
  (void)first.contains(key);
  second[key] = val; // no-warning: The second lookup is on a different container.
}

void countThenContains(std::map<int, int> &m, int key) {
  (void)m.count(key);    // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void containsThanContains(std::map<int, int> &m, int key) {
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void subscriptThenContains(std::map<int, int> &m, int key) {
  (void)m[key];          // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void allLookupsAreMentioned(std::map<int, int> &m, int key) {
  (void)m.at(key);             // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  (void)m.contains(key);       // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.count(key);          // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.find(key);           // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.size();              // no-warning: Not a lookup call.
  m[key] = 1;                  // CHECK-MESSAGES: :[[@LINE]]:3: note: next lookup here
  (void)m[key];                // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.count(key);          // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.emplace(key, 2);
  (void)m.emplace(key, 2);
  (void)m.try_emplace(key, 3);
  (void)m.try_emplace(key, 3);
  (void)m.insert({key, 4});
  (void)m.insert({key, 4});
}

void aliasesAreNotTracked(std::map<int, int> &m, int key) {
  auto &alias = m;
  (void)alias[key];
  (void)m.contains(key); // FIXME: "alias" is considered a separate object.
}

void mutationBetweenLookups(std::map<int, int> &m, int key) {
  extern std::map<int, int> *global_map;
  // FP: We ignore mutations between the lookups.
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  global_map = &m;
  escape(m);
  (void)m.contains(key); // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
}

void nested(std::map<int, int> &m, int key) {
  (void)m.contains(key); // no-warning: This lookup is in a different stack frame.
}
void onlyLocalLookupsAreConsidered(std::map<int, int> &m, int key) {
  (void)m.contains(key);
  nested(m, key);
}

void lookupsWithinMacrosAreIgnored(std::map<int, int> &m, int key) {
  myassert(m.count(key) == 0);
  myassert(m.count(key) == 0);
  myassert(m.count(key) == 0);
  (void)m.count(key); // no-warining: This is just the first lookup that counts.
}

void lookupsWithinMacrosAreIgnored2(std::map<int, int> &m, int key) {
  (void)m.count(key); // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  myassert(m.count(key) == 0);
  myassert(m.count(key) == 0);
  myassert(m.count(key) == 0);
  m[key] = 10; // CHECK-MESSAGES: :[[@LINE]]:3: note: next lookup here
}

void sideffectsAreIgnoredInKeyExpr(std::map<int, int> &m, int n) {
  // FIXME: This is a FP. We should probably ignore expressions with definite sideffects.
  (void)m.contains(rng(++n));   // CHECK-MESSAGES: :[[@LINE]]:9: warning: possibly redundant container lookups
  (void)m.contains(rng(++n));   // CHECK-MESSAGES: :[[@LINE]]:9: note: next lookup here
  (void)m.contains(rng(n + 1)); // no-warning: This uses a different lookup key.
}
