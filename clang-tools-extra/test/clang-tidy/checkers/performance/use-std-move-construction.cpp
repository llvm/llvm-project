// RUN: %check_clang_tidy %s performance-use-std-move %t -std=c++11,c++14,c++17,c++20,c++23
// RUN: %clang -std=c++11 -pedantic-errors -fsyntax-only -nostdinc++ -isystem %clang_tidy_headers/std %t.cpp

#include <initializer_list>
#include <utility>

struct Movable {
  Movable();
  Movable(const Movable &);
  Movable(Movable &&);
  Movable &operator=(const Movable &);
  Movable &operator=(Movable &&);
  void touch() const;
};
void consume(Movable);
void consumeTwice(Movable, Movable);
void borrow(const Movable &);
bool again();

void localInitialization() {
  Movable source;
  source.touch();
  Movable target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Movable target(std::move(source));
  target.touch();
}

void valueParameter(Movable source) {
  consume(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: consume(source);
}

void rvalueReference(Movable &&source) {
  Movable target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Movable target(source);
}

using Alias = Movable;
void parenthesizedAlias(Alias source) {
  Alias target((source));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Alias target((std::move(source)));
}

void lastOfSeveralCopies(Movable source) {
  consume(source);
  Movable target = source;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Movable target = std::move(source);
}

void branches(bool condition, Movable source) {
  if (condition) {
    consume(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'source' could be moved here [performance-use-std-move]
  } else {
    source.touch();
    consume(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: 'source' could be moved here [performance-use-std-move]
  }
}

void usedLater(Movable source) {
  consume(source);
  source.touch();
}
void usedAfterBranch(bool condition, Movable source) {
  if (condition)
    consume(source);
  source.touch();
}
void unorderedUses(Movable source) { consumeTwice(source, source); }
void loop(Movable source) {
  while (again())
    consume(source);
}
void rangeLoop(Movable source) {
  int values[2] = {};
  for (int value : values)
    consume(source);
}
void backwardJump(Movable source) {
repeat:
  consume(source);
  if (again())
    goto repeat;
}

void referenceAlias(Movable source) {
  Movable &alias = (source);
  consume(source);
  alias.touch();
}
void constReferenceAlias(Movable source) {
  const Movable &alias = source;
  consume(source);
  alias.touch();
}
void pointerAlias(Movable source) {
  const Movable *alias = &source;
  consume(source);
  alias->touch();
}
void escapedReference(Movable source) {
  borrow(source);
  consume(source);
}
void referenceCapture(Movable source) {
  auto later = [&source] { source.touch(); };
  consume(source);
  later();
}
void copyCapture(Movable source) {
  auto later = [source] { source.touch(); };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: auto later = [source] { source.touch(); };
  later();
}
void implicitCopyCapture(Movable source) {
  auto later = [=] { source.touch(); };
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: auto later = [=] { source.touch(); };
  later();
}

void alreadyMoved(Movable source) { consume(std::move(source)); }
void constSource(const Movable source) { consume((source)); }
void referenceSource(Movable &source) { consume(source); }
void builtInType(int source) { int target = source; }

struct CopyOnly {
  CopyOnly(const CopyOnly &);
};
void consume(CopyOnly);
void copyOnly(CopyOnly source) { consume(source); }
struct DeletedMove {
  Movable field;
  DeletedMove(const DeletedMove &);
  DeletedMove(DeletedMove &&) = delete;
};
void consume(DeletedMove);
void deletedMove(DeletedMove source) { consume(source); }
struct PrivateMove {
  PrivateMove(const PrivateMove &);

private:
  PrivateMove(PrivateMove &&);
};
void consume(PrivateMove);
void privateMove(PrivateMove source) { consume(source); }
struct Trivial {
  int value;
};
void consume(Trivial);
void trivialCopy(Trivial source) { consume(source); }
struct TrivialMove {
  TrivialMove(const TrivialMove &);
  TrivialMove(TrivialMove &&) = default;
};
void consume(TrivialMove);
void trivialMove(TrivialMove source) {
  consume(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: consume(source);
}

struct ReferenceHolder {
  ReferenceHolder(const Movable &);
};
void referenceConstructor(Movable source) { ReferenceHolder target(source); }
void referenceReceiver(Movable source) { borrow(source); }

Movable global;
void globalSource() { Movable target(global); }
void staticSource() {
  static Movable source;
  Movable target(source);
}
void threadLocalSource() {
  thread_local Movable source;
  Movable target(source);
}
void externalSource() {
  extern Movable source;
  Movable target(source);
}

#define COPY(source) consume((source))
void macro(Movable source) {
  COPY(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: COPY(source);
}
void macroBeforeUse(Movable source) {
  COPY(source);
  source.touch();
}

Movable returnLocal() {
  Movable source;
  return source;
}

void overloaded(Movable);
void overloaded(Movable &&);
void overloadedReceiver(Movable source) {
  overloaded(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: overloaded(source);
}

template <class T> void templateCopy(T source) {
  T target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: T target(source);
}
template void templateCopy<Movable>(Movable);
template void templateCopy<CopyOnly>(CopyOnly);
template void templateCopy<TrivialMove>(TrivialMove);
template <class T> void forwardingReference(T &&source) {
  consume(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: consume(source);
}
template void forwardingReference<Movable>(Movable &&);
template void forwardingReference<Movable &>(Movable &);

struct Derived : Movable {};
void copyBase(Derived source) { Movable target(source); }

struct AmbiguousMove {
  AmbiguousMove(const AmbiguousMove &);
  AmbiguousMove(AmbiguousMove &&, int = 0);
  AmbiguousMove(AmbiguousMove &&, double = 0);
};
void ambiguousMove(AmbiguousMove source) { AmbiguousMove target(source); }

struct CheapCopy {
  CheapCopy(const CheapCopy &) = default;
  CheapCopy(CheapCopy &&);
};
void cheapCopy(CheapCopy source) { CheapCopy target(source); }

void shorterDestinationLifetime(Movable source) {
  {
    Movable target(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'source' could be moved here [performance-use-std-move]
    // CHECK-FIXES: Movable target(source);
  }
}

void shorterForInitializerLifetime(Movable source) {
  for (Movable target(source); again();)
    break;
  // CHECK-MESSAGES: :[[@LINE-2]]:23: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: for (Movable target(source); again();)
}

void shorterImplicitScopeLifetime(Movable source, bool condition) {
  if (condition)
    Movable target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: Movable target(source);
}

struct MemberOwner {
  Movable field;
  MemberOwner(const MemberOwner &);
  MemberOwner(MemberOwner &&);
};
void memberAlias(MemberOwner source) {
  const Movable &alias = source.field;
  MemberOwner target(source);
  alias.touch();
}

struct ReferenceOwner {
  ReferenceOwner(const ReferenceOwner &);
  ReferenceOwner(ReferenceOwner &&);
  const ReferenceOwner &reference() const;
};
void returnedAlias(ReferenceOwner source) {
  const ReferenceOwner &alias = source.reference();
  ReferenceOwner target(source);
  alias.reference();
}

void constRvalueReference(const Movable &&source) { consume(source); }

struct ExplicitMove {
  ExplicitMove(const ExplicitMove &);
  explicit ExplicitMove(ExplicitMove &&);
};
void explicitMove(ExplicitMove source) {
  ExplicitMove target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ExplicitMove target(std::move(source));
}
void explicitCopyInitialization(ExplicitMove source) { ExplicitMove target = source; }

struct MoveOnly {
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&);
};
void consume(MoveOnly);
void movedMoveOnly(MoveOnly source) { consume(std::move(source)); }

void assignmentAlias(Movable &target, Movable source) {
  const Movable &alias = source;
  target = source;
  alias.touch();
}

void assignmentUnorderedUses(Movable &target, Movable source) {
  consumeTwice(target = source, source);
  // CHECK-FIXES: consumeTwice(target = source, source);
}

void discardedReferenceResult(ReferenceOwner source) {
  source.reference();
  ReferenceOwner target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ReferenceOwner target(std::move(source));
}

struct PointerOwner {
  PointerOwner(const PointerOwner &);
  PointerOwner(PointerOwner &&);
  const PointerOwner *pointer() const;
};
void returnedPointerAlias(PointerOwner source) {
  const PointerOwner *alias = source.pointer();
  PointerOwner target(source);
  alias->pointer();
}

void assignedPointerAlias(PointerOwner source) {
  const PointerOwner *alias;
  alias = source.pointer();
  PointerOwner target(source);
  alias->pointer();
}

void conditionalAlias(bool condition, PointerOwner source,
                      const PointerOwner *other) {
  const PointerOwner *alias = condition ? source.pointer() : other;
  PointerOwner target(source);
  alias->pointer();
}

void exceptionUse(Movable source) {
  try {
    consume(source);
  } catch (...) {
    source.touch();
  }
}

struct ExtraArguments {
  ExtraArguments(const ExtraArguments &, int = 0);
  ExtraArguments(ExtraArguments &&);
};
void explicitExtraArgument(ExtraArguments source) {
  ExtraArguments target(source, 1); // The move cannot accept both arguments.
}
void defaultExtraArgument(ExtraArguments source) {
  ExtraArguments target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ExtraArguments target(std::move(source));
}

struct VariadicMove {
  VariadicMove(const VariadicMove &);
  VariadicMove(VariadicMove &&, ...);
};
void variadicMove(VariadicMove source) { VariadicMove target(source); }

struct NontrivialDestructor {
  int value;
  NontrivialDestructor(const NontrivialDestructor &) = default;
  NontrivialDestructor(NontrivialDestructor &&) = default;
  ~NontrivialDestructor();
};
void cheapCopyWithDestructor(NontrivialDestructor source) {
  NontrivialDestructor target(source);
}

Movable conditionalRvalueReference(Movable &&source, bool condition) {
  return condition ? source : Movable{};
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: return condition ? source : Movable{};
}

struct ListOverload {
  ListOverload(const ListOverload &);
  ListOverload(ListOverload &&);
  ListOverload(std::initializer_list<int>);
  operator int() &&;
};
void changedListOverload(ListOverload source) {
  ListOverload target{source};
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: 'source' could be moved here [performance-use-std-move]
  // CHECK-FIXES: ListOverload target{source};
}

struct ConstMove {
  ConstMove(const ConstMove &);
  ConstMove(const ConstMove &&);
};
void constMove(ConstMove source) { ConstMove target(source); }

struct VolatileMove {
  VolatileMove(const VolatileMove &);
  VolatileMove(volatile VolatileMove &&);
};
void volatileMove(VolatileMove source) { VolatileMove target(source); }
