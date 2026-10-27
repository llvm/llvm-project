// RUN: %check_clang_tidy %s modernize-loop-convert %t

struct Container {
  struct iterator {
    ~iterator() {}
    int &operator*() const;
    iterator &operator++();
    bool operator!=(const iterator &) const;
  };
  iterator begin();
  iterator end();
};

void f(Container &C) {
  for (Container::iterator It = C.begin(); It != C.end(); ++It)
    (void)*It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : C)
  // CHECK-FIXES-NEXT: (void)It;

  for (Container::iterator It = C.begin(); It != (C.end()); ++It)
    (void)*It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : C)
  // CHECK-FIXES-NEXT: (void)It;

  for (Container::iterator It = C.begin(); C.end() != It; ++It)
    (void)*It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : C)
  // CHECK-FIXES-NEXT: (void)It;

  for (Container::iterator It = C.begin(), E = C.end(); It != E; ++It)
    (void)*It;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & It : C)
  // CHECK-FIXES-NEXT: (void)It;
}

void negative(Container &C, Container &D) {
  for (Container::iterator It = C.begin(); It != D.end(); ++It)
    (void)*It;
}
