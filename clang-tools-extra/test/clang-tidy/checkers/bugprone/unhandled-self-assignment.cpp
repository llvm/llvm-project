// RUN: %check_clang_tidy %s bugprone-unhandled-self-assignment %t -- -- -fno-delayed-template-parsing

namespace std {

template <class T>
void swap(T &x, T &y) {
}

template <class T>
T &&move(T &x) {
}

template <class T>
class unique_ptr {
};

template <class T>
class shared_ptr {
};

template <class T>
class weak_ptr {
};

template <class T>
class auto_ptr {
};

} // namespace std

void assert(int expression){};

///////////////////////////////////////////////////////////////////
/// Test cases correctly caught by the check.

class PtrField {
public:
  PtrField &operator=(const PtrField &object);

private:
  int *p;
};

PtrField &PtrField::operator=(const PtrField &object) {
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
  // ...
  return *this;
}

// Class with an inline operator definition.
class InlineDefinition {
public:
  InlineDefinition &operator=(const InlineDefinition &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:21: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  int *p;
};

class UniquePtrField {
public:
  UniquePtrField &operator=(const UniquePtrField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:19: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  std::unique_ptr<int> p;
};

class SharedPtrField {
public:
  SharedPtrField &operator=(const SharedPtrField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:19: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  std::shared_ptr<int> p;
};

class WeakPtrField {
public:
  WeakPtrField &operator=(const WeakPtrField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:17: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  std::weak_ptr<int> p;
};

class AutoPtrField {
public:
  AutoPtrField &operator=(const AutoPtrField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:17: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  std::auto_ptr<int> p;
};

// Class with C array field.
class CArrayField {
public:
  CArrayField &operator=(const CArrayField &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:16: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  int array[256];
};

// Make sure to not ignore cases when the operator definition calls
// a copy constructor of another class.
class CopyConstruct {
public:
  CopyConstruct &operator=(const CopyConstruct &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:18: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    WeakPtrField a;
    WeakPtrField b(a);
    // ...
    return *this;
  }

private:
  int *p;
};

// Make sure to not ignore cases when the operator definition calls
// a copy assignment operator of another class.
class AssignOperator {
public:
  AssignOperator &operator=(const AssignOperator &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:19: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    a.operator=(object.a);
    // ...
    return *this;
  }

private:
  int *p;
  WeakPtrField a;
};

class NotSelfCheck {
public:
  NotSelfCheck &operator=(const NotSelfCheck &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:17: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    if (&object == this->doSomething()) {
      // ...
    }
    return *this;
  }

  void *doSomething() {
    return p;
  }

private:
  int *p;
};

template <class T>
class TemplatePtrField {
public:
  TemplatePtrField<T> &operator=(const TemplatePtrField<T> &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:24: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  T *p;
};

template <class T>
class TemplateCArrayField {
public:
  TemplateCArrayField<T> &operator=(const TemplateCArrayField<T> &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:27: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    // ...
    return *this;
  }

private:
  T p[256];
};

// Other template class's constructor is called inside a declaration.
template <class T>
class WrongTemplateCopyAndMove {
public:
  WrongTemplateCopyAndMove<T> &operator=(const WrongTemplateCopyAndMove<T> &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:32: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    TemplatePtrField<T> temp;
    TemplatePtrField<T> temp2(temp);
    return *this;
  }

private:
  T *p;
};

// https://bugs.llvm.org/show_bug.cgi?id=44499
class Foo2;
template <int a>
bool operator!=(Foo2 &, Foo2 &) {
  class Bar2 {
    Bar2 &operator=(const Bar2 &other) {
      // CHECK-MESSAGES: [[@LINE-1]]:11: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
      p = other.p;
      return *this;
    }

    int *p;
  };
}

///////////////////////////////////////////////////////////////////
/// Test cases correctly ignored by the check.

// Self-assignment is checked using the equality operator.
class SelfCheck1 {
public:
  SelfCheck1 &operator=(const SelfCheck1 &object) {
    if (this == &object)
      return *this;
    // ...
    return *this;
  }

private:
  int *p;
};

class SelfCheck2 {
public:
  SelfCheck2 &operator=(const SelfCheck2 &object) {
    if (&object == this)
      return *this;
    // ...
    return *this;
  }

private:
  int *p;
};

// Self-assignment is checked using the inequality operator.
class SelfCheck3 {
public:
  SelfCheck3 &operator=(const SelfCheck3 &object) {
    if (this != &object) {
      // ...
    }
    return *this;
  }

private:
  int *p;
};

class SelfCheck4 {
public:
  SelfCheck4 &operator=(const SelfCheck4 &object) {
    if (&object != this) {
      // ...
    }
    return *this;
  }

private:
  int *p;
};

template <class T>
class TemplateSelfCheck {
public:
  TemplateSelfCheck<T> &operator=(const TemplateSelfCheck<T> &object) {
    if (&object != this) {
      // ...
    }
    return *this;
  }

private:
  T *p;
};

// https://bugs.llvm.org/show_bug.cgi?id=44499
class Foo;
template <int a>
bool operator!=(Foo &, Foo &) {
  class Bar {
    Bar &operator=(const Bar &other) {
      if (this != &other) {
      }
      return *this;
    }

    int *p;
  };
}

// There is no warning if the copy assignment operator gets the object by value.
class PassedByValue {
public:
  PassedByValue &operator=(PassedByValue object) {
    // ...
    return *this;
  }

private:
  int *p;
};

// User-defined swap method calling std::swap inside.
class CopyAndSwap1 {
public:
  CopyAndSwap1 &operator=(const CopyAndSwap1 &object) {
    CopyAndSwap1 temp(object);
    doSwap(temp);
    return *this;
  }

private:
  int *p;

  void doSwap(CopyAndSwap1 &object) {
    using std::swap;
    swap(p, object.p);
  }
};

// User-defined swap method used with passed-by-value parameter.
class CopyAndSwap2 {
public:
  CopyAndSwap2 &operator=(CopyAndSwap2 object) {
    doSwap(object);
    return *this;
  }

private:
  int *p;

  void doSwap(CopyAndSwap2 &object) {
    using std::swap;
    swap(p, object.p);
  }
};

// Copy-and-swap method is used but without creating a separate method for it.
class CopyAndSwap3 {
public:
  CopyAndSwap3 &operator=(const CopyAndSwap3 &object) {
    CopyAndSwap3 temp(object);
    std::swap(p, temp.p);
    return *this;
  }

private:
  int *p;
};

template <class T>
class TemplateCopyAndSwap {
public:
  TemplateCopyAndSwap<T> &operator=(const TemplateCopyAndSwap<T> &object) {
    TemplateCopyAndSwap<T> temp(object);
    std::swap(p, temp.p);
    return *this;
  }

private:
  T *p;
};

// Move semantics is used on a temporary copy of the object.
class CopyAndMove1 {
public:
  CopyAndMove1 &operator=(const CopyAndMove1 &object) {
    CopyAndMove1 temp(object);
    *this = std::move(temp);
    return *this;
  }

private:
  int *p;
};

// There is no local variable for the temporary copy.
class CopyAndMove2 {
public:
  CopyAndMove2 &operator=(const CopyAndMove2 &object) {
    *this = CopyAndMove2(object);
    return *this;
  }

private:
  int *p;
};

template <class T>
class TemplateCopyAndMove {
public:
  TemplateCopyAndMove<T> &operator=(const TemplateCopyAndMove<T> &object) {
    TemplateCopyAndMove<T> temp(object);
    *this = std::move(temp);
    return *this;
  }

private:
  T *p;
};

// There is no local variable for the temporary copy.
template <class T>
class TemplateCopyAndMove2 {
public:
  TemplateCopyAndMove2<T> &operator=(const TemplateCopyAndMove2<T> &object) {
    *this = std::move(TemplateCopyAndMove2<T>(object));
    return *this;
  }

private:
  T *p;
};

// We should not catch move assignment operators.
class MoveAssignOperator {
public:
  MoveAssignOperator &operator=(MoveAssignOperator &&object) {
    // ...
    return *this;
  }

private:
  int *p;
};

// We ignore copy assignment operators without user-defined implementation.
class DefaultOperator {
public:
  DefaultOperator &operator=(const DefaultOperator &object) = default;

private:
  int *p;
};

class DeletedOperator {
public:
  DeletedOperator &operator=(const DefaultOperator &object) = delete;

private:
  int *p;
};

class ImplicitOperator {
private:
  int *p;
};

// Check ignores those classes which has no any pointer or array field.
class TrivialFields {
public:
  TrivialFields &operator=(const TrivialFields &object) {
    // ...
    return *this;
  }

private:
  int m;
  float f;
  double d;
  bool b;
};

// There is no warning when the class calls another assignment operator on 'this'
// inside the copy assignment operator's definition.
class AssignIsForwarded {
public:
  AssignIsForwarded &operator=(const AssignIsForwarded &object) {
    operator=(object.p);
    return *this;
  }

  AssignIsForwarded &operator=(int *pp) {
    if (p != pp) {
      delete p;
      p = new int(*pp);
    }
    return *this;
  }

private:
  int *p;
};

// Assertion is a valid way to say that self-assignment is not expected to happen.
class AssertGuard {
public:
  AssertGuard &operator=(const AssertGuard &object) {
    assert(this != &object);
    // ...
    return *this;
  }

private:
  int *p;
};

// Make sure we don't catch this operator=() as a copy assignment operator.
// Note that RHS has swapped template arguments.
template <typename Ty, typename Uy>
class NotACopyAssignmentOperator {
  Ty *Ptr1;
  Uy *Ptr2;

public:
  NotACopyAssignmentOperator& operator=(const NotACopyAssignmentOperator<Uy, Ty> &RHS) {
    Ptr1 = RHS.getUy();
    Ptr2 = RHS.getTy();
    return *this;
  }

  Ty *getTy() const { return Ptr1; }
  Uy *getUy() const { return Ptr2; }
};

///////////////////////////////////////////////////////////////////
/// Test cases which should be caught by the check.

// TODO: handle custom pointers.
template <class T>
class custom_ptr {
};

class CustomPtrField {
public:
  CustomPtrField &operator=(const CustomPtrField &object) {
    // ...
    return *this;
  }

private:
  custom_ptr<int> p;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////
/// False positives: These are self-assignment safe, but they don't use any of the three patterns.

class ArrayCopy {
public:
  ArrayCopy &operator=(const ArrayCopy &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:14: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    for (int i = 0; i < 256; i++)
      array[i] = object.array[i];
    return *this;
  }

private:
  int array[256];
};

class GetterSetter {
public:
  GetterSetter &operator=(const GetterSetter &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:17: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    setValue(object.getValue());
    return *this;
  }

  int *getValue() const { return value; }

  void setValue(int *newPtr) {
    int *pTmp(newPtr ? new int(*newPtr) : nullptr);
    std::swap(value, pTmp);
    delete pTmp;
  }

private:
  int *value;
};

class CustomSelfCheck {
public:
  CustomSelfCheck &operator=(const CustomSelfCheck &object) {
    // CHECK-MESSAGES: [[@LINE-1]]:20: warning: operator=() does not handle self-assignment properly [bugprone-unhandled-self-assignment]
    if (index != object.index) {
      // ...
    }
    return *this;
  }

private:
  int *value;
  int index;
};
