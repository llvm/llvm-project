// RUN: %check_clang_tidy %s performance-use-std-move %t -std=c++11,c++14,c++17,c++20,c++23
// RUN: %clang -std=c++11 -pedantic-errors -fsyntax-only -nostdinc++ -isystem %clang_tidy_headers/std %t.cpp

#include <utility>

struct Value {
  int value;
  Value();
  Value(const Value &);
  Value(Value &&) noexcept;
  Value &operator=(const Value &) noexcept;
  Value &operator=(Value &&) noexcept;
  [[clang::reinitializes]] void refresh() noexcept;
  [[clang::reinitializes]] void throwingRefresh();
  void mutate() noexcept;
  Value &target() noexcept;
};
void consume(Value);
void consumeTwo(Value, Value);
void read(int);
void retain(const Value &);
void inspect(Value * __attribute__((noescape)));
bool again();

void localAliasUsedBeforeCopy(Value source) {
  const Value &alias = (source);
  read(alias.value);
  Value target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target(std::move(source));
}

void transitiveAlias(Value source) {
  const Value &first = source;
  const Value &second = first;
  read(second.value);
  Value target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target(std::move(source));
}

void transitiveAliasUsedLater(Value source) {
  const Value &first = source;
  const Value &second = first;
  Value target(source);
  read(second.value);
}

void aliasEscapesBeforeCopy(Value source) {
  const Value &alias = source;
  retain(alias);
  Value target(source);
}

void transientAddress(Value source) {
  inspect(&source);
  Value target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target(std::move(source));
}

void pointerAlias(Value source) {
  Value *alias = &source;
  read(alias->value);
  Value target(source);
}

void freshObjectInLoop(int count) {
  for (int i = 0; i < count; ++i) {
    Value source;
    Value target(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
    // CHECK-FIXES: Value target(std::move(source));
  }
}

void restoredInLoop(Value source) {
  while (again()) {
    consume(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
    // CHECK-FIXES: consume(source);
    source = Value{};
  }
}

void annotationRestores(Value source) {
  consume(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  source.refresh();
  read(source.value);
}

void throwingRestoration(Value source) {
  try {
    consume(source);
    source.throwingRefresh();
  } catch (...) {
    read(source.value);
  }
}

void mutationIsNotRestoration(Value source) {
  consume(source);
  source.mutate();
  read(source.value);
}

void restorationOnOnlyOnePath(bool condition, Value source) {
  consume(source);
  if (condition)
    source = Value{};
  read(source.value);
}

void restorationOnBothPaths(bool condition, Value source) {
  consume(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  if (condition)
    source = Value{};
  else
    source.refresh();
  read(source.value);
}

void orderedComma(Value source) {
  consume((read(source.value), source));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: consume((read(source.value), source));
}

void nestedOrderedComma(Value source) {
  Value target((read(0), (read(source.value), source)));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target((read(0), (read(source.value), std::move(source))));
}

void commaBeforeUse(Value source) {
  consume((read(source.value), source));
  read(source.value);
}

Value conditionalCopy(bool condition, Value source) {
  return condition ? source : Value{};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: return condition ? source : Value{};
}

void capture(Value source) {
  auto closure = [source] { return source.value; };
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: auto closure = [source] { return source.value; };
}

void defaultCapture(Value source) {
  auto closure = [=] { return source.value; };
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: auto closure = [=] { return source.value; };
}

void captureBeforeUse(Value source) {
  auto closure = [source] { return source.value; };
  read(source.value);
}

void referenceCapture(Value source) {
  auto closure = [&source] { return source.value; };
  Value target(source);
  read(closure());
}

void lambdaLocal() {
  auto closure = [](Value source) {
    Value target(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
    // CHECK-FIXES: Value target(std::move(source));
  };
}

void heapCopy(Value source) {
  Value *target = new Value(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value *target = new Value(source);
  delete target;
}

void explicitTemporary(Value source) {
  retain(Value(source));
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: retain(Value(source));
}

struct Aggregate { Value field; };
void aggregate(Value source) {
  Aggregate target{source};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Aggregate target{source};
}

void listInitialization(Value source) {
  Value target{source};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target{std::move(source)};
}

void unevaluatedUse(Value source) {
  Value target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target(std::move(source));
  (void)sizeof(source);
  (void)noexcept(source.refresh());
}

void selfAssignment(Value source) {
  source = source;
  // CHECK-FIXES: source = source;
}

void selfAssignmentThroughAlias(Value source) {
  Value &alias = source;
  alias = source;
}

void transitiveSelfAssignment(Value source) {
  Value &first = source;
  Value &second = first;
  second = source;
}

void castSelfAssignment(Value source) {
  static_cast<Value &>(source) = source;
}
void castAliasSelfAssignment(Value source) {
  Value &alias = source;
  static_cast<Value &>(alias) = source;
}
void sourceDependentReceiver(Value source) {
  source.target() = source; // target() could return source itself.
}
void conditionalSelfAssignment(Value source, Value &other, bool condition) {
  (condition ? source : other) = source;
}

struct PointerHolder {
  PointerHolder(const Value *);
};
void escapedThroughConstructor(Value source) {
  PointerHolder holder(&source);
  Value target(source);
}

struct VariadicHolder {
  VariadicHolder(int, ...);
};
void escapedThroughVariadicConstructor(Value source) {
  VariadicHolder holder(0, &source);
  Value target(source);
}

struct ReferenceAggregate {
  const Value &reference;
};
void escapedThroughAggregate(Value source) {
  ReferenceAggregate holder{source};
  Value target(source);
}
void pointerArithmeticEscape(Value source) {
  const Value *alias = &source + 0;
  Value target(source);
}
void commaReferenceEscape(Value source) {
  const Value &alias = (read(0), source);
  Value target(source);
}
void thrownPointerEscape(Value source) {
  try {
    throw &source;
  } catch (const Value *pointer) {
    // Even without a later named use, the handler could retain the pointer.
    inspect(const_cast<Value *>(pointer));
  }
  Value target(source);
}

struct RvalueAssignment {
  RvalueAssignment &operator=(const RvalueAssignment &) &;
  RvalueAssignment &operator=(RvalueAssignment &&) &&;
};
void wrongReceiver(RvalueAssignment &target, RvalueAssignment source) {
  target = source;
}

struct NewException {
  int value;
  NewException(const NewException &) noexcept;
  NewException(NewException &&);
};
void introducedHandler(NewException source) {
  try {
    NewException target(source);
  } catch (...) {
    read(source.value);
  }
}
void introducedExceptionWithoutHandler(NewException source) {
  NewException target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: NewException target(source);
}

#define COPY_TWICE(x) consumeTwo(x, x)
void repeatedMacro(Value source) { COPY_TWICE(source); }

struct ConversionOwner { Value field; };
void fieldGap(ConversionOwner source) { consume(source.field); }
void pointerGap(Value *source) { consume(*source); }

struct ValueHolder {
  ValueHolder(Value);
};
void byValueConstructor(Value source) {
  ValueHolder target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'source' could be moved here
  // CHECK-FIXES: ValueHolder target(source);
}
void constructorAfterLoop() {
  Value source;
  while (again())
    source.mutate();
  ValueHolder target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'source' could be moved here
  // CHECK-FIXES: ValueHolder target(source);
}
void guardedConstructor(Value source) {
  if (source.value)
    ValueHolder target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: 'source' could be moved here
  // CHECK-FIXES: ValueHolder target(source);
}
void fieldDestination(Value source) {
  Aggregate target;
  target.field = source;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'source' could be moved here
  // CHECK-FIXES: target.field = std::move(source);
}
struct MemberInitializers {
  Value first;
  Value second;
  MemberInitializers(Value source) : first(source), second(source) {}
};

void dependentRestoration(Value source) {
  consume(source);
  source = Value(source);
  read(source.value);
}
void restorationThroughAlias(Value source) {
  Value &alias = source;
  consume(source); // Alias assignment is not currently a restoration proof.
  alias = Value{};
  read(source.value);
}
