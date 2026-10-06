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

  // Expression arguments pass through verbatim.
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

void bracedRange(std::vector<int> &V, const int *Begin, const int *End,
                 std::string &S, const char *Chars, size_type N) {
  // A braced list that does not fit the element type list-initializes the
  // same kind of temporary through a multi-argument constructor.
  V = {Begin, End};
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(Begin, End);

  S = {Chars, N};
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; use 'assign'
  // CHECK-FIXES: S.assign(Chars, N);
}

void initializerList(std::vector<int> &V) {
  V = std::vector<int>{1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; assign the initializer list directly to reuse the existing storage
  // CHECK-FIXES: V = {1, 2, 3};

  V = std::vector<int>({4, 5});
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; assign the initializer list directly
  // CHECK-FIXES: V = {4, 5};
}

void count(std::vector<int> &V, size_type N) {
  V = std::vector<int>(N);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize' to reuse the existing storage
  // CHECK-FIXES: V.clear(); V.resize(N);

  // Not a statement of its own, so there is no room for two statements:
  // diagnosed without a fix-it.
  if (N)
    V = std::vector<int>(N);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize'
  // CHECK-FIXES: V = std::vector<int>(N);

  // The count would be evaluated after 'clear()': diagnosed without a fix-it.
  V = std::vector<int>(getCount());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize'
  // CHECK-FIXES: V = std::vector<int>(getCount());
}

void direct(std::vector<int> &V, const std::vector<int> &Other,
            std::vector<int> &&Temp) {
  V = std::vector<int>(Other);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; assign the source directly to reuse the existing storage
  // CHECK-FIXES: V = Other;

  V = std::vector<int>(std::move(Temp));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; assign the source directly
  // CHECK-FIXES: V = std::move(Temp);
}

void destinations(Holder &H, size_type N) {
  H.Items = std::vector<int>(N, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: H.Items.assign(N, 1);

  *H.Ptr = std::vector<int>(N, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: (*H.Ptr).assign(N, 1);

  getVector() = std::vector<int>(N, 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: getVector().assign(N, 2);

  H.Items = std::vector<int>(N);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize'
  // CHECK-FIXES: H.Items.clear(); H.Items.resize(N);

  // The destination would be evaluated twice: diagnosed without a fix-it.
  getVector() = std::vector<int>(N);
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize'
  // CHECK-FIXES: getVector() = std::vector<int>(N);
}

void singleStatementBody(std::vector<int> &V, size_type N) {
  if (N)
    V = std::vector<int>(N, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(N, 0);
}

void aliasing(std::vector<int> &V, Holder &H) {
  // Iterators into the destination are diagnosed but not rewritten: 'assign'
  // must not be given iterators into the container it replaces.
  V = std::vector<int>(V.begin() + 1, V.end());
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V = std::vector<int>(V.begin() + 1, V.end());

  // A count is computed before 'assign' runs, so it may mention the
  // destination.
  V = std::vector<int>(V.size(), 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: V.assign(V.size(), 0);

  H.Items = std::vector<int>(H.Count, 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: H.Items.assign(H.Count, 1);

  // With 'clear()' first, the count would be read from an emptied container.
  H.Items = std::vector<int>(H.Items.size());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: inefficient assignment from a temporary 'std::vector'; use 'clear' and 'resize'
  // CHECK-FIXES: H.Items = std::vector<int>(H.Items.size());
}

void valueUsed(std::vector<int> &V, std::vector<int> &W, size_type N) {
  // 'assign' does not yield the container, so the value of the assignment has
  // to be discarded for the rewrite: diagnosed without a fix-it.
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

  S = std::string(Other);
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: inefficient assignment from a temporary 'std::basic_string'; assign the source directly
  // CHECK-FIXES: S = Other;

  // A single argument of another type is a conversion, which is a different
  // matter: no diagnostic.
  S = std::string(Chars);
}

void notDiagnosed(std::vector<int> &V, size_type N, std::allocator<int> Alloc) {
  // Default construction releases the storage; 'clear()' would keep it.
  V = std::vector<int>();
  V = std::vector<int>{};

  // No temporary container is involved.
  V = {};
  V = {1, 2};

  // An explicitly passed allocator has no counterpart in 'assign'.
  V = std::vector<int>(N, 0, Alloc);

  // Initializations are copy-elided since C++17; nothing to save.
  std::vector<int> Init = std::vector<int>(N, 0);
}

void shrinkToFit(std::vector<int> &V, Holder &H) {
  // Copying a container into itself is the idiom for trimming its capacity;
  // there the temporary is the point, so there is no diagnostic.
  V = std::vector<int>(V);
  H.Items = std::vector<int>(H.Items);
}

struct Derived : std::vector<int> {
  using std::vector<int>::operator=;
};

void derived(Derived &D, size_type N) {
  // The destination's type differs from the temporary's: no diagnostic.
  D = std::vector<int>(N, 0);
}

template <typename T>
void dependent(std::vector<T> &V, size_type N) {
  // Type-dependent: no diagnostic, including in instantiations.
  V = std::vector<T>(N, T());
}
void instantiate(std::vector<int> &V) { dependent(V, 3); }

#define ASSIGN_FILL(Dest, Count) Dest = std::vector<int>(Count, 0)
void macro(std::vector<int> &V) {
  // Diagnosed, but no fix-it: rewriting a macro expansion is unsafe.
  ASSIGN_FILL(V, 3);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: inefficient assignment from a temporary 'std::vector'; use 'assign'
  // CHECK-FIXES: ASSIGN_FILL(V, 3);
}
