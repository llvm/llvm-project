// RUN: %check_clang_tidy %s performance-use-std-move %t

// Definitions used in the tests
// -----------------------------

namespace std {
template<class T> struct remove_reference { typedef T type; };
template<class T> struct remove_reference<T&> { typedef T type; };
template<class T> struct remove_reference<T&&> { typedef T type; };
template< class T >
constexpr typename remove_reference<T>::type&& move( T&& t ) noexcept;
}

struct NonTrivialMoveAssign {
  NonTrivialMoveAssign() = default;
  NonTrivialMoveAssign(const NonTrivialMoveAssign&) = default;

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

template<class T>
void use(T&) {}

// Check moving from various reference/pointer type
// ------------------------------------------------

void ConvertibleNonTrivialMoveAssign(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

void NonProfitableNonTrivialMoveAssignPointer(NonTrivialMoveAssign*& target, NonTrivialMoveAssign* source) {
  // No message expected, moving is possible but non profitable for pointer.
  target = source;
}

void ConvertibleNonTrivialMoveAssignFromLValue(NonTrivialMoveAssign& target, NonTrivialMoveAssign&& source) {
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

// Check moving already moved values
// ---------------------------------

void NonConvertibleNonTrivialMoveAssignAlreadyMoved(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // No message expected, it's already moved
  target = std::move(source);
}

void NonConvertibleNonTrivialMoveAssignFromLValueAlreadyMoved(NonTrivialMoveAssign& target, NonTrivialMoveAssign&& source) {
  // No message expected, it's already moved
  target = std::move(source);
}

// Check moving within different context
// -------------------------------------

struct SomeRecord {
void ConvertibleNonTrivialMoveAssignWithinMethod(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}
};

auto ConvertibleNonTrivialMoveAssignWithinLambda = [](NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
};

void SomeFunction(NonTrivialMoveAssign source0, NonTrivialMoveAssign const &source1) {

auto NonConvertibleNonTrivialMoveAssignWithinLambdaAsCaptureByRef = [&](NonTrivialMoveAssign& target) {
  // No message expected, cannot move a non-local variable.
  target = source0;
  // No message expected, cannot move a non-local variable.
  target = source1;
};

auto ConvertibleNonTrivialMoveAssignWithinLambdaAsCapture = [=](NonTrivialMoveAssign& target) {
  // No message expected, cannot move a non-local variable.
  target = source0;
  // No message expected, cannot move a non-local variable.
  target = source1;
};

}

void ConvertibleNonTrivialMoveAssignShadowing(NonTrivialMoveAssign& target, NoMoveAssign source) {
  {
    NonTrivialMoveAssign source;
    // CHECK-MESSAGES: [[@LINE+1]]:14: warning: 'source' could be moved here [performance-use-std-move]
    target = source;
  }
}

void ConvertibleNonTrivialMoveAssignShadowed(NoMoveAssign& target, NonTrivialMoveAssign source) {
  {
    NoMoveAssign source;
    // No message expected, `source' refers to a non-movable variable.
    target = source;
  }
}

#define ASSIGN(x, y) x = y
void ConvertibleNonTrivialMoveAssignWithinMacro(NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  // No message expected, assignment within a macro.
  ASSIGN(target, source);
}

template<class T>
void ConvertibleNonTrivialMoveAssignWithTemplateSource(NonTrivialMoveAssign& target, T source) {
  // No message expected, type of source cannot be matched against `target's type.
  target = source;
}

template<class T>
void ConvertibleNonTrivialMoveAssignWithTemplateTarget(T& target, NonTrivialMoveAssign source) {
  // No message expected, type of target cannot be matched against `source's type.
  target = source;
}

// Check moving from various storage
// ---------------------------------

void NonConvertibleNonTrivialMoveAssignFromLocal(NonTrivialMoveAssign& target) {
  const NonTrivialMoveAssign source;
  // No message, moving a const-qualified value is not valid.
  target = source;
}

void NonConvertibleNonTrivialMoveAssignFromConst(NonTrivialMoveAssign& target) {
  NonTrivialMoveAssign source;
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

void NonConvertibleNonTrivialMoveAssignFromStatic(NonTrivialMoveAssign& target) {
  static NonTrivialMoveAssign source;
  // No message, the lifetime of `source' does not end with the scope of the function.
  target = source;
}

struct NonConvertibleNonTrivialMoveAssignFromMember {
  NonTrivialMoveAssign source;
  void NonConvertibleNonTrivialMoveAssignFromStatic(NonTrivialMoveAssign& target) {
    // No message, `source' is not a local variable.
    target = source;
  }
};

void NonConvertibleNonTrivialMoveAssignFromExtern(NonTrivialMoveAssign& target) {
  extern NonTrivialMoveAssign source;
  // No message, the lifetime of `source' does not end with the scope of the function.
  target = source;
}

void NonConvertibleNonTrivialMoveAssignFromTLS(NonTrivialMoveAssign& target) {
  thread_local NonTrivialMoveAssign source;
  // No message, the lifetime of `source' does not end with the scope of the function.
  target = source;
}

NonTrivialMoveAssign global_source;
void NonConvertibleNonTrivialMoveAssignToGlobal(NonTrivialMoveAssign& target) {
  // No message, the lifetime of `source' does not end with the scope of the function.
  target = global_source;
}


// Check moving to various storage
// -------------------------------

void ConvertibleNonTrivialMoveAssignToStatic(NonTrivialMoveAssign source) {
  static NonTrivialMoveAssign target;
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

struct ConvertibleNonTrivialMoveAssignToMember {
  NonTrivialMoveAssign target;
  void NonConvertibleNonTrivialMoveAssignFromStatic(NonTrivialMoveAssign source) {
    // CHECK-MESSAGES: [[@LINE+1]]:14: warning: 'source' could be moved here [performance-use-std-move]
    target = source;
  }
};

void ConvertibleNonTrivialMoveAssignToExtern(NonTrivialMoveAssign source) {
  extern NonTrivialMoveAssign target;
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

void ConvertibleNonTrivialMoveAssignToTLS(NonTrivialMoveAssign source) {
  thread_local NonTrivialMoveAssign target;
  // CHECK-MESSAGES: [[@LINE+1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  target = source;
}

NonTrivialMoveAssign global_target;
void ConvertibleNonTrivialMoveAssignToGlobal(NonTrivialMoveAssign source) {
  // CHECK-MESSAGES: [[@LINE+1]]:19: warning: 'source' could be moved here [performance-use-std-move]
  global_target = source;
}

void NonConvertibleNonTrivialMoveAssignRValue(NonTrivialMoveAssign& target, NonTrivialMoveAssign const& source) {
  // No message expected, moving a reference is invalid there.
  target = source;
}

void NonProfitableTrivialMoveAssign(TrivialMoveAssign& target, TrivialMoveAssign source) {
  // No message expected, moving is possible but pedantic.
  target = source;
}

// Check moving in presence of control flow or use
// -----------------------------------------------

void ConvertibleNonTrivialMoveAssignWithBranching(bool cond, NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  if(cond) {
    // CHECK-MESSAGES: [[@LINE+1]]:14: warning: 'source' could be moved here [performance-use-std-move]
    target = source;
  }
}

void NonConvertibleNonTrivialMoveAssignWithBranchingAndUse(bool cond, NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  if(cond) {
    // No message expected, moving would make use invalid.
    target = source;
  }
  use(source);
}

void ConvertibleNonTrivialMoveAssignBothBranches(bool cond, NonTrivialMoveAssign& target, NonTrivialMoveAssign source) {
  if(cond) {
    // CHECK-MESSAGES: [[@LINE+1]]:14: warning: 'source' could be moved here [performance-use-std-move]
    target = source;
  }
  else {
    source.stuff();
    // CHECK-MESSAGES: [[@LINE+1]]:14: warning: 'source' could be moved here [performance-use-std-move]
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

// Check moving for invalid / non profitable type or operation
// -----------------------------------------------------------

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
