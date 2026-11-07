// RUN: %check_clang_tidy %s performance-inefficient-container-assignment %t

#include <deque>
#include <forward_list>
#include <list>
#include <string>
#include <utility>
#include <vector>

using size_type = std::vector<int>::size_type;

std::vector<int> &getVector();
size_type getCount();

struct Holder {
  std::vector<int> Items;
  std::vector<int> *Ptr;
  size_type Count;
};

void fill(std::vector<int> &V, size_type N) {
  V = std::vector<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign' to reuse the existing storage [performance-inefficient-container-assignment]
  // CHECK-FIXES: V.assign(N, 0);

  V = std::vector<int>(N + 1, -1);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(N + 1, -1);
}

void range(std::vector<int> &V, const int *Begin, const int *End,
           const std::vector<int> &Other) {
  V = std::vector<int>(Begin, End);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Begin, End);

  V = std::vector<int>(Other.begin(), Other.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Other.begin(), Other.end());
}

void convertingRange(std::vector<std::string> &V, const char *const *Begin,
                     const char *const *End, char **Argv, int Argc) {
  V = std::vector<std::string>(Begin, End);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Begin, End);

  V = std::vector<std::string>(Argv + 1, Argv + Argc);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Argv + 1, Argv + Argc);
}

void bracedRange(std::vector<int> &V, const int *Begin, const int *End,
                 std::string &S, const char *Chars, size_type N) {
  V = {Begin, End};
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Begin, End);

  S = {Chars, N};
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; use 'assign'
  // CHECK-FIXES: S.assign(Chars, N);
}

void destinations(Holder &H, size_type N) {
  H.Items = std::vector<int>(N, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: H.Items.assign(N, 1);

  *H.Ptr = std::vector<int>(N, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: H.Ptr->assign(N, 1);

  getVector() = std::vector<int>(N, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: getVector().assign(N, 2);
}

struct Member {
  std::vector<int> Items;

  void reset(size_type N) {
    Items = std::vector<int>(N, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
    // CHECK-FIXES: Items.assign(N, 0);
  }
};

void singleStatementBody(std::vector<int> &V, size_type N) {
  if (N)
    V = std::vector<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(N, 0);
}

void parenthesized(std::vector<int> &V, size_type N) {
  (V = std::vector<int>(N, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: (V.assign(N, 0));
}

void evaluationOrder(std::vector<std::vector<int>> &VS, std::vector<int> *Arr,
                     size_type I) {
  // The assignment evaluates the temporary before the destination, whereas
  // 'assign' evaluates the destination first.
  VS[I++] = std::vector<int>(I + 1, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: VS[I++] = std::vector<int>(I + 1, 0);

  VS[I] = std::vector<int>(I++, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: VS[I] = std::vector<int>(I++, 0);

  Arr[I] = std::vector<int>(getCount(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: Arr[I].assign(getCount(), 0);

  VS.back() = std::vector<int>(I, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: VS.back().assign(I, 0);
}

void aliasing(std::vector<int> &V, Holder &H, size_type N) {
  // 'assign' must not be given iterators into the container it replaces.
  V = std::vector<int>(V.begin() + 1, V.end());
  V = std::vector<int>(V.end() - N, V.end());

  // A count is computed before 'assign' runs.
  V = std::vector<int>(V.size(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(V.size(), 0);

  H.Items = std::vector<int>(H.Count, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: H.Items.assign(H.Count, 1);
}

void valueUsed(std::vector<int> &V, std::vector<int> &W, size_type N) {
  W = (V = std::vector<int>(N, 0));
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: W = (V = std::vector<int>(N, 0));
}

std::vector<int> &returned(std::vector<int> &V, size_type N) {
  return V = std::vector<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: return V = std::vector<int>(N, 0);
}

void otherContainers(std::deque<int> &D, std::list<int> &L,
                     std::forward_list<int> &F, size_type N) {
  D = std::deque<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::deque'; use 'assign'
  // CHECK-FIXES: D.assign(N, 0);

  L = std::list<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::list'; use 'assign'
  // CHECK-FIXES: L.assign(N, 0);

  F = std::forward_list<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::forward_list'; use 'assign'
  // CHECK-FIXES: F.assign(N, 0);
}

void strings(std::string &S, const std::string &Other, const char *Chars,
             size_type N) {
  S = std::string(N, ' ');
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; use 'assign'
  // CHECK-FIXES: S.assign(N, ' ');

  S = std::string(Chars, N);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; use 'assign'
  // CHECK-FIXES: S.assign(Chars, N);

  S = std::string(Other, 1, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; use 'assign'
  // CHECK-FIXES: S.assign(Other, 1, 2);

  // A conversion rather than a copy.
  S = std::string(Chars);
}

struct NoCopyAssign {
  NoCopyAssign(const NoCopyAssign &);
  NoCopyAssign &operator=(const NoCopyAssign &) = delete;
};

struct ConstMember {
  const int Value;
};

struct ExplicitFromInt {
  explicit ExplicitFromInt(int);
};

void elementRequirements(std::vector<NoCopyAssign> &V,
                         const NoCopyAssign &Value, const NoCopyAssign *Begin,
                         const NoCopyAssign *End, std::vector<ConstMember> &C,
                         const ConstMember &Member,
                         std::vector<ExplicitFromInt> &E, const int *Ints,
                         size_type N) {
  // 'assign' needs assignable elements, which constructing the temporary does
  // not.
  V = std::vector<NoCopyAssign>(N, Value);
  V = std::vector<NoCopyAssign>(Begin, End);
  C = std::vector<ConstMember>(N, Member);
  E = std::vector<ExplicitFromInt>(Ints, Ints + N);
}

void notDiagnosed(std::vector<int> &V, size_type N,
                  std::allocator<int> Alloc) {
  // Default construction releases the storage; 'clear()' would keep it.
  V = std::vector<int>();
  V = std::vector<int>{};

  // No temporary container is involved.
  V = {};
  V = {1, 2};

  // An explicitly passed allocator has no counterpart in 'assign'.
  V = std::vector<int>(N, 0, Alloc);

  // Initializations are copy-elided since C++17.
  std::vector<int> Init = std::vector<int>(N, 0);
}

struct Derived : std::vector<int> {
  using std::vector<int>::operator=;
};

void derived(Derived &D, size_type N) {
  D = std::vector<int>(N, 0);
}

template <typename T>
void dependent(std::vector<T> &V, size_type N) {
  V = std::vector<T>(N, T());
}
void instantiate(std::vector<int> &V) { dependent(V, 3); }

#define ASSIGN_FILL(Dest, Count) Dest = std::vector<int>(Count, 0)
void macro(std::vector<int> &V) {
  ASSIGN_FILL(V, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: ASSIGN_FILL(V, 3);
}
