// RUN: %check_clang_tidy %s altera-id-dependent-backward-branch %t -- -header-filter=.* "--" -cl-std=CLC++1.0 -c
// RUN: %check_clang_tidy -std=c++20-or-later %s altera-id-dependent-backward-branch %t -- -header-filter=.* --

unsigned long get_local_id(unsigned);
int foo(int);

#ifndef __OPENCL_CPP_VERSION__
namespace std {
typedef decltype(sizeof(0)) size_t;

template <class E>
class initializer_list {
public:
  typedef const E *iterator;
  typedef const E *const_iterator;
  typedef size_t size_type;

private:
  iterator Begin;
  size_type Size;

  constexpr initializer_list(const_iterator Begin, size_type Size)
      : Begin(Begin), Size(Size) {}

public:
  constexpr initializer_list() : Begin(nullptr), Size(0) {}
  constexpr size_type size() const { return Size; }
  constexpr const_iterator begin() const { return Begin; }
  constexpr const_iterator end() const { return Begin + Size; }
};
} // namespace std
#endif

void error() {
  // ==== Conditional Expressions ====
  int accumulator = 0;
  for (int i = 0; i < get_local_id(0); i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  int j = 0;
  while (j < get_local_id(0)) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < get_local_id(0));
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to ID function call and may cause performance degradation [altera-id-dependent-backward-branch]

  // ==== Assignments ====
  int ThreadID = get_local_id(0);

  while (j < ThreadID) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: assignment of ID-dependent variable ThreadID
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < ThreadID);
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to variable reference to 'ThreadID' and may cause performance degradation [altera-id-dependent-backward-branch]
  // CHECK-NOTES: :[[@LINE-12]]:3: note: assignment of ID-dependent variable ThreadID

  struct { int IDDepField; } Example;
  Example.IDDepField = get_local_id(0);

  for (int i = 0; i < Example.IDDepField; i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: assignment of ID-dependent field IDDepField
    accumulator++;
  }

  while (j < Example.IDDepField) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-10]]:3: note: assignment of ID-dependent field IDDepField
    accumulator++;
  }

  do {
    accumulator++;
  } while (j < Example.IDDepField);
  // CHECK-NOTES: :[[@LINE-1]]:12: warning: backward branch (do loop) is ID-dependent due to member reference to 'IDDepField' and may cause performance degradation [altera-id-dependent-backward-branch]
  // CHECK-NOTES: :[[@LINE-18]]:3: note: assignment of ID-dependent field IDDepField

  // ==== Inferred Assignments ====
  int ThreadID2 = ThreadID * 2;

  for (int i = 0; i < ThreadID2; i++) {
    // CHECK-NOTES: :[[@LINE-1]]:19: warning: backward branch (for loop) is ID-dependent due to variable reference to 'ThreadID2' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-4]]:3: note: inferred assignment of ID-dependent value from ID-dependent variable ThreadID
    accumulator++;
  }

  struct {
    int FieldFromVar;
    int FieldFromField;
  } InferredField;
  InferredField.FieldFromVar = ThreadID * 2;
  while (j < InferredField.FieldFromVar) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to member reference to 'FieldFromVar' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-6]]:5: note: inferred assignment of ID-dependent member from ID-dependent variable ThreadID
    accumulator++;
  }

  InferredField.FieldFromField = Example.IDDepField;
  while (j < InferredField.FieldFromField) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to member reference to 'FieldFromField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-12]]:5: note: inferred assignment of ID-dependent member from ID-dependent member IDDepField
    accumulator++;
  }

  int ThreadIDFromField = Example.IDDepField;
  while (j < ThreadIDFromField) {
    // CHECK-NOTES: :[[@LINE-1]]:10: warning: backward branch (while loop) is ID-dependent due to variable reference to 'ThreadIDFromField' and may cause performance degradation [altera-id-dependent-backward-branch]
    // CHECK-NOTES: :[[@LINE-3]]:3: note: inferred assignment of ID-dependent value from ID-dependent member IDDepField
    accumulator++;
  }

  // ==== Unused Inferred Assignments ====
  int UnusedThreadID = Example.IDDepField; // OK: not used in any loops

  struct { int IDDepField; } UnusedStruct;
  UnusedStruct.IDDepField = ThreadID * 2; // OK: not used in any loops
}

void success() {
  int accumulator = 0;

  for (int i = 0; i < 1000; i++) {
    if (i < get_local_id(0)) {
      accumulator++;
    }
  }

  // ==== Regression tests: ordinary values are not ID-dependent ====
  int Value = 0;
  int OtherValue = 1;
  Value = OtherValue;
  while (Value < OtherValue) {
    ++Value;
  }

  int ComputedValue = foo(0);
  while (Value < ComputedValue) {
    ++Value;
  }

  int InferredValue = OtherValue * 2;
  for (int i = 0; i < InferredValue; ++i) {
    accumulator += i;
  }

  struct {
    int OrdinaryField;
    int OtherOrdinaryField;
  } OrdinaryStruct;
  OrdinaryStruct.OrdinaryField = foo(0);
  while (Value < OrdinaryStruct.OrdinaryField) {
    ++Value;
  }

  int FromOrdinaryField = OrdinaryStruct.OrdinaryField;
  while (Value < FromOrdinaryField) {
    ++Value;
  }

  OrdinaryStruct.OtherOrdinaryField = OtherValue;
  while (Value < OrdinaryStruct.OtherOrdinaryField) {
    ++Value;
  }

#define ASSIGN_ORDINARY_ID_DEPENDENT_TEST(lhs, rhs) lhs = rhs
  int MacroAssigned = 0;
  ASSIGN_ORDINARY_ID_DEPENDENT_TEST(MacroAssigned, OtherValue);
  while (MacroAssigned < OtherValue) {
    ++MacroAssigned;
  }
#undef ASSIGN_ORDINARY_ID_DEPENDENT_TEST

  typedef int IntTypedef;
  using IntAlias = int;
  IntTypedef TypedefValue = OtherValue;
  IntAlias AliasValue = TypedefValue;
  while (AliasValue < OtherValue) {
    ++AliasValue;
  }

  const int ConstValue = OtherValue;
  volatile int VolatileValue = OtherValue;
  int &ValueRef = Value;
  ValueRef = ConstValue;
  while (ValueRef < VolatileValue) {
    ++ValueRef;
  }

  auto OrdinaryLambda = [&Value, OtherValue]() {
    int LambdaValue = OtherValue;
    while (Value < LambdaValue) {
      ++Value;
    }
  };
  OrdinaryLambda();

#ifndef __OPENCL_CPP_VERSION__
  for (int ChunkSize : {1, 2, 3}) {
    for (int i = 0; i < 500; i += ChunkSize) {
      static_cast<void>(i);
    }
  }
#endif
}

template<char... STOP>
void gh55408(char const input[], int pos) {
  while (((input[pos] != STOP) && ...));
}

template <typename T>
void ordinary_template(T Value) {
  T OtherValue = Value;
  while (OtherValue < Value) {
    ++OtherValue;
  }
}

template <typename T>
void dependent_template() {
  T Value = T();
  T OtherValue = Value;
  while (OtherValue < Value) {
    ++OtherValue;
  }
}

#ifndef __OPENCL_CPP_VERSION__
template <typename T>
concept HasValue = requires(T Value) {
  Value + 1;
};

template <HasValue T>
void ordinary_constrained_template(T Value) {
  T OtherValue = Value;
  while (OtherValue < Value) {
    ++OtherValue;
  }
}
#endif
