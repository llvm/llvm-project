// RUN: %check_clang_tidy %s readability-deleted-default %t -- -- -fno-ms-compatibility

class NoDefault {
public:
  NoDefault() = delete;
  NoDefault(NoDefault &&Other) = delete;
  NoDefault(const NoDefault &Other) = delete;
};

class MissingEverything {
public:
  MissingEverything() = default;
  // CHECK-MESSAGES: warning: default constructor is explicitly defaulted but implicitly deleted, probably because a non-static data member or a base class is lacking a default constructor; definition can either be removed or explicitly deleted [readability-deleted-default]
  MissingEverything(MissingEverything &&Other) = default;
  // CHECK-MESSAGES: warning: move constructor is explicitly defaulted but implicitly deleted, probably because a non-static data member or a base class is neither copyable nor movable; definition can either be removed or explicitly deleted [readability-deleted-default]
  MissingEverything(const MissingEverything &Other) = default;
  // CHECK-MESSAGES: warning: copy constructor is explicitly defaulted but implicitly deleted, probably because a non-static data member or a base class is not copyable; definition can either be removed or explicitly deleted [readability-deleted-default]
  MissingEverything &operator=(MissingEverything &&Other) = default;
  // CHECK-MESSAGES: warning: move assignment operator is explicitly defaulted but implicitly deleted, probably because a base class or a non-static data member is not assignable, e.g. because the latter is marked 'const'; definition can either be removed or explicitly deleted [readability-deleted-default]
  MissingEverything &operator=(const MissingEverything &Other) = default;
  // CHECK-MESSAGES: warning: copy assignment operator is explicitly defaulted but implicitly deleted, probably because a base class or a non-static data member is not assignable, e.g. because the latter is marked 'const'; definition can either be removed or explicitly deleted [readability-deleted-default]

private:
  NoDefault ND;
};

class NotAssignable {
public:
  NotAssignable(NotAssignable &&Other) = default;
  NotAssignable(const NotAssignable &Other) = default;
  NotAssignable &operator=(NotAssignable &&Other) = default;
  // CHECK-MESSAGES: warning: move assignment operator is explicitly defaulted but implicitly deleted
  NotAssignable &operator=(const NotAssignable &Other) = default;
  // CHECK-MESSAGES: warning: copy assignment operator is explicitly defaulted but implicitly deleted

private:
  const int I = 0;
};

class Movable {
public:
  Movable() = default;
  Movable(Movable &&Other) = default;
  Movable(const Movable &Other) = delete;
  Movable &operator=(Movable &&Other) = default;
  Movable &operator=(const Movable &Other) = delete;
};

class NotCopyable {
public:
  NotCopyable(NotCopyable &&Other) = default;
  NotCopyable(const NotCopyable &Other) = default;
  // CHECK-MESSAGES: warning: copy constructor is explicitly defaulted but implicitly deleted
  NotCopyable &operator=(NotCopyable &&Other) = default;
  NotCopyable &operator=(const NotCopyable &Other) = default;
  // CHECK-MESSAGES: warning: copy assignment operator is explicitly defaulted but implicitly deleted
private:
  Movable M;
};

template <typename T> class Templated {
public:
  // No warning here, it is a templated class.
  Templated() = default;
  Templated(Templated &&Other) = default;
  Templated(const Templated &Other) = default;
  Templated &operator=(Templated &&Other) = default;
  Templated &operator=(const Templated &Other) = default;

  class InnerTemplated {
  public:
    // This class is not in itself templated, but we still don't have warning.
    InnerTemplated() = default;
    InnerTemplated(InnerTemplated &&Other) = default;
    InnerTemplated(const InnerTemplated &Other) = default;
    InnerTemplated &operator=(InnerTemplated &&Other) = default;
    InnerTemplated &operator=(const InnerTemplated &Other) = default;

  private:
    T TVar;
  };

  class InnerNotTemplated {
  public:
    // This one could technically have warnings, but currently doesn't.
    InnerNotTemplated() = default;
    InnerNotTemplated(InnerNotTemplated &&Other) = default;
    InnerNotTemplated(const InnerNotTemplated &Other) = default;
    InnerNotTemplated &operator=(InnerNotTemplated &&Other) = default;
    InnerNotTemplated &operator=(const InnerNotTemplated &Other) = default;

  private:
    int I;
  };

private:
  const T TVar{};
};

int FunctionWithInnerClass() {
  class InnerNotAssignable {
  public:
    InnerNotAssignable &operator=(InnerNotAssignable &&Other) = default;
    // CHECK-MESSAGES: warning: move assignment operator is explicitly defaulted but implicitly deleted
  private:
    const int I = 0;
  };
  return 1;
};

template <typename T>
int TemplateFunctionWithInnerClass() {
  class InnerNotAssignable {
  public:
    InnerNotAssignable &operator=(InnerNotAssignable &&Other) = default;
  private:
    const T TVar{};
  };
  return 1;
};

void Foo() {
  Templated<const int> V1;
  Templated<int>::InnerTemplated V2;
  Templated<float>::InnerNotTemplated V3;
  TemplateFunctionWithInnerClass<int>();
}
