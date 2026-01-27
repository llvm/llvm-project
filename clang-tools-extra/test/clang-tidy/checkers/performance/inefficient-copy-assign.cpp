// RUN: %check_clang_tidy %s performance-inefficient-copy-assign %t

struct NonTrivialMoveAssign {
  NonTrivialMoveAssign& operator=(const NonTrivialMoveAssign&);
  NonTrivialMoveAssign& operator=(NonTrivialMoveAssign&&);
  NonTrivialMoveAssign& operator+=(const NonTrivialMoveAssign&);
  NonTrivialMoveAssign& operator+=(NonTrivialMoveAssign&&);
  void stuff();
};

struct TrivialMoveAssign {
  TrivialMoveAssign& operator=(const TrivialMoveAssign&);
  TrivialMoveAssign& operator=(TrivialMoveAssign&&) = default;
};

struct NoMoveAssign {
  NoMoveAssign& operator=(const NoMoveAssign&);
  NoMoveAssign& operator=(NoMoveAssign&&) = delete;
};

void ConvertibleNonTrivialMoveAssign(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // CHECK-MESSAGES: [[@LINE+1]]:{{[0-9]*}}: warning: 'source' could be moved here [performance-inefficient-copy-assign]
  target = source;
}

void NonProfitableTrivialMoveAssign(TrivialMoveAssign& target, TrivialMoveAssign source) {
  // No message expected, moving is possible but pedantic.
  target = source;
}

void ConvertibleNonTrivialMoveAssignWithBranching(bool cond, NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  if(cond) {
    // CHECK-MESSAGES: [[@LINE+1]]:{{[0-9]*}}: warning: 'source' could be moved here [performance-inefficient-copy-assign]
    target = source;
  }
}

void ConvertibleNonTrivialMoveAssignBothBranches(bool cond, NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  if(cond) {
    // CHECK-MESSAGES: [[@LINE+1]]:{{[0-9]*}}: warning: 'source' could be moved here [performance-inefficient-copy-assign]
    target = source;
  }
  else {
    source.stuff();
    // CHECK-MESSAGES: [[@LINE+1]]:{{[0-9]*}}: warning: 'source' could be moved here [performance-inefficient-copy-assign]
    target = source;
  }
}

void NonConvertibleNonTrivialMoveAssignWithExtraUse(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // No message expected, moving would make the call to `stuff' invalid.
  target = source;
  source.stuff();
}

void NonConvertibleNonTrivialMoveAssignInLoop(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // No message expected, moving would make the next iteration invalid.
  for(int i = 0; i < 10; ++i)
    target = source;
}

void NonConvertibleNonTrivialMoveUpdateAssign(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // No message currently expected, we only consider assignment.
  target += source;
}

void NonProfitableTrivialTypeAssign(int& target, int source) {
  // No message needed, it's correct to move but pedantic.
  target = source;
}

void InvalidMoveAssign(NoMoveAssign& target, NoMoveAssign source) {
  // No message expected, moving is deleted.
  target = source;
}
