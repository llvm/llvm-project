// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoDeleteChecker -verify %s

#include "mock-types.h"

void *memcpy(void *dst, const void *src, unsigned int size);
void *malloc(unsigned int size);
void free(void *);

namespace WTF {

  template <typename T>
  class Vector {
  public:
    ~Vector() { destory(); }

    void append(const T& v)
    {
      if (m_size >= m_capacity)
        grow(m_capacity * 2);
      new (m_buffer + m_size) T();
      m_buffer[m_size] = v;
      m_size++;
    }

    void shrink(unsigned newSize)
    {
      unsigned currentSize = m_size;
      while (currentSize > newSize) {
        --currentSize;
        m_buffer[currentSize].~T();
      }
      m_size = currentSize;
    }

  private:
    void grow(unsigned newCapacity) {
      T* newBuffer = static_cast<T*>(malloc(sizeof(T) * newCapacity));
      memcpy(newBuffer, m_buffer, sizeof(T) * m_size);
      destory();
      m_buffer = newBuffer;
      m_capacity = newCapacity;
    }

    void destory() {
      if (!m_buffer)
        return;
      for (unsigned i = 0; i < m_size; ++i)
        m_buffer[i].~T();
      free(m_buffer);
      m_buffer = nullptr;
    }

    T* m_buffer { nullptr };
    unsigned m_size { 0 };
    unsigned m_capacity { 0 };
  };

} // namespace WTF

using WTF::Vector;

void someFunction();
RefCountable* [[clang::annotate_type("webkit.nodelete")]] safeFunction();

void functionWithoutNoDeleteAnnotation() {
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafe() {
  someFunction(); // expected-warning{{A function 'callsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

int* [[clang::annotate_type("webkit.nodelete")]] createsInt() {
  return new int;
}

void [[clang::annotate_type("webkit.nodelete")]] destroysInt(int* number) {
  delete number; // expected-warning{{A function 'destroysInt' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

struct IntPoint {
  int x { 0 };
  int y { 0 };
};

IntPoint* [[clang::annotate_type("webkit.nodelete")]] createsIntPoint() {
  return new IntPoint[2];
}

void [[clang::annotate_type("webkit.nodelete")]] destroysIntPoint(IntPoint* point) {
  delete[] point; // expected-warning{{A function 'destroysIntPoint' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] callOperatorDelete(int* number) {
  ::operator delete(number); // expected-warning{{A function 'callOperatorDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafeWithSuppress();

[[clang::suppress]] void callsUnsafeWithSuppress() {
  someFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] callsNoDeleteFunction() {
  callsUnsafeWithSuppress();
}

#define EXPORT_IMPORT __attribute__((visibility("default")))
EXPORT_IMPORT unsigned [[clang::annotate_type("webkit.nodelete")]] safeFunctionWithAttr();

void [[clang::annotate_type("webkit.nodelete")]] callsSafeWithAttribute() {
  unsigned r = safeFunctionWithAttr();
}

void [[clang::annotate_type("webkit.nodelete")]] callsSafe() {
  safeFunction();
}

void [[clang::annotate_type("webkit.nodelete")]] declWithNoDelete();
void declWithNoDelete() {
  someFunction(); // expected-warning{{A function 'declWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}
void defWithNoDelete();
void [[clang::annotate_type("webkit.nodelete")]] defWithNoDelete() {
  someFunction(); // expected-warning{{A function 'defWithNoDelete' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(Ref<RefCountable> t) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 't' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(unsigned safe, Ref<RefCountable> unsafe) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 'unsafe' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithUnsafeParam(Ref<RefCountable> unsafe, unsigned safe) {
  // expected-warning@-1{{A function 'funncWithUnsafeParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 'unsafe' which could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] funncWithSafeParam(Ref<RefCountable>& safe1, Ref<RefCountable>* safe2) {
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafeInDoWhile() {
  do {
    someFunction(); // expected-warning{{A function 'callsUnsafeInDoWhile' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  } while(0);
}

void [[clang::annotate_type("webkit.nodelete")]] callsUnsafeInIf(bool safe) {
  if (safe)
    safeFunction();
  else
    someFunction(); // expected-warning{{A function 'callsUnsafeInIf' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] declaresUnsafeVar(bool safe) {
  if (safe) {
    auto* t = safeFunction();
  } else {
    RefPtr<RefCountable> t = safeFunction();
    // expected-warning@-1{{A function 'declaresUnsafeVar' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
}

void [[clang::annotate_type("webkit.nodelete")]] declaresVarInIf(bool safe) {
  if (RefPtr<RefCountable> t = safeFunction()) {
    // expected-warning@-1{{A function 'declaresVarInIf' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    t->method();
  }
}

template <typename T>
struct TemplatedClass {
  void [[clang::annotate_type("webkit.nodelete")]] methodCallsUnsafe(T* t) {
    t->method(); // expected-warning{{A function 'methodCallsUnsafe' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  void [[clang::annotate_type("webkit.nodelete")]] methodCallsSafe(T* t) {
    t->trivial();
  }
};

using TemplatedToRefCountable = TemplatedClass<RefCountable>;
void useTemplatedToRefCountable() {
  TemplatedToRefCountable c;
  c.methodCallsUnsafe(nullptr);
  c.methodCallsSafe(nullptr);
}

class WeakRefCountable : public CanMakeWeakPtr<WeakRefCountable> {
public:
  static Ref<WeakRefCountable> create();

  ~WeakRefCountable();

  void ref() { m_refCount++; }
  void deref() {
    m_refCount--;
    if (!m_refCount)
      delete this;
  }

private:
  WeakRefCountable();

  unsigned m_refCount { 0 };
};

class SomeClass {
public:

  void [[clang::annotate_type("webkit.nodelete")]] someMethod();
  void [[clang::annotate_type("webkit.nodelete")]] unsafeMethod() {
    someFunction(); // expected-warning{{A function 'unsafeMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  void [[clang::annotate_type("webkit.nodelete")]] safeMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] someVirtualMethod();
  virtual void [[clang::annotate_type("webkit.nodelete")]] unsafeVirtualMethod() {
    someFunction(); // expected-warning{{A function 'unsafeVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  virtual void [[clang::annotate_type("webkit.nodelete")]] safeVirtualMethod() {
    safeFunction();
  }

  static void [[clang::annotate_type("webkit.nodelete")]] someStaticMethod();
  static void [[clang::annotate_type("webkit.nodelete")]] unsafeStaticMethod() {
    someFunction(); // expected-warning{{A function 'unsafeStaticMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  static void [[clang::annotate_type("webkit.nodelete")]] safeStaticMethod() {
    safeFunction();
  }

  virtual void [[clang::annotate_type("webkit.nodelete")]] anotherVirtualMethod();

  void [[clang::annotate_type("webkit.nodelete")]] setObj(RefCountable* obj) {
    m_obj = obj; // expected-warning{{A function 'setObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] swapObj(RefPtr<RefCountable>&& obj) {
    m_obj.swap(obj);
  }

  void [[clang::annotate_type("webkit.nodelete")]] assignObj(Ref<RefCountable>&& obj);

  void [[clang::annotate_type("webkit.nodelete")]] clearObj(RefCountable* obj) {
    m_obj = nullptr; // expected-warning{{A function 'clearObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeArg(WeakRefCountable&& unused) {
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeArgPtr(RefPtr<RefCountable>&& unused) {
  }

  enum class E : unsigned char { V1, V2 };
  bool [[clang::annotate_type("webkit.nodelete")]] deposeArgEnum() {
    E&& e = E::V1;
    return e != E::V2;
  }

  void [[clang::annotate_type("webkit.nodelete")]] deposeLocal() {
    RefPtr<RefCountable> obj = std::move(m_obj); // expected-warning{{A function 'deposeLocal' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  RefPtr<RefCountable> [[clang::annotate_type("webkit.nodelete")]] copyRefPtr() {
    return m_obj;
  }

  Ref<WeakRefCountable> [[clang::annotate_type("webkit.nodelete")]] copyRef() {
    return *m_weakObj.get();
  }

  RefPtr<WeakRefCountable> [[clang::annotate_type("webkit.nodelete")]] getWeakPtr() {
    return m_weakObj.get();
  }

  WeakRefCountable* [[clang::annotate_type("webkit.nodelete")]] useWeakPtr() {
    auto* localWeak = m_weakObj.get();
    return localWeak;
  }

private:
  RefPtr<RefCountable> m_obj;
  Ref<RefCountable> m_ref;
  WeakPtr<WeakRefCountable> m_weakObj;
};


void SomeClass::assignObj(Ref<RefCountable>&& obj) {
  m_obj = std::move(obj);
   // expected-warning@-1{{A function 'assignObj' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

class IntermediateClass : public SomeClass {
  void anotherVirtualMethod() override;
};

class DerivedClass : public IntermediateClass {
  void anotherVirtualMethod() override {
    someFunction(); // expected-warning{{A function 'anotherVirtualMethod' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
};

template <class Type>
class Base {
public:
  virtual unsigned foo() const = 0;
};

template <class Type>
class Derived : public Base<Type> {
public:
  virtual unsigned foo() const { return 0; }
};

struct Data {
  static Ref<Data> [[clang::annotate_type("webkit.nodelete")]] create() {
    return adoptRef(*new Data);
  }

  static Ref<Data> [[clang::annotate_type("webkit.nodelete")]] create(double) {
    return adoptRef(*new Data(RefCountable::create()->next()));
    // expected-warning@-1{{A function 'create' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  static Data* [[clang::annotate_type("webkit.nodelete")]] create(int) {
    return adoptRef(new Data); // expected-warning{{A function 'create' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  static Ref<Data> create(const char*) {
    return std::move(adoptRef(*new Data));
  }

  void ref() {
    ++refCount;
  }

  void deref() {
    --refCount;
    if (!refCount)
      delete this;
  }

  virtual void doSomething() { }

  virtual void [[clang::annotate_type("webkit.nodelete")]] virtualWork() { }

  int a[3] { 0 };
  
protected:
  Data() = default;
  Data(RefCountable*) { }

private:
  unsigned refCount { 0 };
};

struct SubData : Data {
  static Ref<SubData> create() {
    return adoptRef(*new SubData);
  }

  void doSomething() override { }

  void virtualWork() override {
    someFunction();
    // expected-warning@-1{{A function 'virtualWork' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

private:
  SubData() = default;
};

void [[clang::annotate_type("webkit.nodelete")]] makeData() {
  RefPtr<Data> constantData[2] = { Data::create() };
  // expected-warning@-1{{A function 'makeData' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  RefPtr<Data> data[] = { Data::create() };
}

void [[clang::annotate_type("webkit.nodelete")]] makeSubData() {
  SubData::create()->doSomething();
  // expected-warning@-1{{A function 'makeSubData' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

struct ObjectWithConstructor {
  ObjectWithConstructor(double x) { }
  ObjectWithConstructor(float x) { }
  ObjectWithConstructor(decltype(nullptr)) { }
  ObjectWithConstructor(void*) { }
  ObjectWithConstructor(int x[3]) { }
  ObjectWithConstructor(void* x[2]) { }
  enum class E { V1, V2 };
  ObjectWithConstructor(E) { }
};

void [[clang::annotate_type("webkit.nodelete")]] makeObjectWithConstructor() {
  ObjectWithConstructor obj1(nullptr);
  ObjectWithConstructor obj2(0.5);
  double x = 0.7;
  ObjectWithConstructor obj3(x);
  int ints[] = { 1, 2, 3 };
  ObjectWithConstructor obj4(ints);
  void* ptrs[] = { nullptr, nullptr };
  ObjectWithConstructor obj5(ptrs);
  ObjectWithConstructor obj6(ObjectWithConstructor::E::V1);
}

struct ObjectWithNonTrivialDestructor {
  ~ObjectWithNonTrivialDestructor();
};

struct Container {
  Ref<Container> create() { return adoptRef(*new Container); }
  void ref() const { refCount++; }
  void deref() const {
    refCount--;
    if (!refCount)
      delete this;
  }

  ObjectWithNonTrivialDestructor obj;

private:
  mutable unsigned refCount { 0 };

  Container() = default;
};

struct SubContainer : public Container {
};

struct OtherContainerBase {
  ObjectWithNonTrivialDestructor obj;
};

struct OtherContainer : public OtherContainerBase {
  Ref<OtherContainer> create() { return adoptRef(*new OtherContainer); }
  void ref() const { refCount++; }
  void deref() const {
    refCount--;
    if (!refCount)
      delete this;
  }

private:
  mutable unsigned refCount { 0 };

  OtherContainer() = default;
};

struct ObjectWithContainers {
  RefPtr<Container> container;
  RefPtr<SubContainer> subContainer;
  RefPtr<OtherContainer> otherContainer;

  void [[clang::annotate_type("webkit.nodelete")]] setContainer(Ref<Container>&& newContainer) {
    container = std::move(newContainer);
    // expected-warning@-1{{A function 'setContainer' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] setSubContainer(Ref<SubContainer>&& newContainer) {
    subContainer = std::move(newContainer);
    // expected-warning@-1{{A function 'setSubContainer' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] setOtherContainer(Ref<OtherContainer>&& newContainer) {
    otherContainer = std::move(newContainer);
    // expected-warning@-1{{A function 'setOtherContainer' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  Vector<Container> containerList;
  Vector<SubContainer> subContainerList;
  Vector<OtherContainer> otherContainerList;

  void [[clang::annotate_type("webkit.nodelete")]] shrinkVector1() {
    containerList.shrink(0);
    // expected-warning@-1{{A function 'shrinkVector1' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] shrinkVector2() {
    subContainerList.shrink(0);
    // expected-warning@-1{{A function 'shrinkVector2' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] shrinkVector3() {
    otherContainerList.shrink(0);
    // expected-warning@-1{{A function 'shrinkVector3' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
};

struct SomeObject {
  void ref() const;
  void deref() const;
  
  void doTrivialWork() { }

  void [[clang::annotate_type("webkit.nodelete")]] deleteItems() {
    delete[] m_items;
    // expected-warning@-1{{A function 'deleteItems' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

private:
  SomeObject* m_items;
};

void [[clang::annotate_type("webkit.nodelete")]] deleteArray(SomeObject* obj) {
  delete[] obj;
  // expected-warning@-1{{A function 'deleteArray' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

template <typename Callback>
void callLambda(Callback callback) {
  callback();
}

template <typename Callback>
void noopWithLambda(Callback) {
}

void [[clang::annotate_type("webkit.nodelete")]] deleteInLambda(SomeObject* someObj) {
  callLambda([&]() {
    // expected-warning@-1{{A function 'deleteInLambda' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    delete someObj;
  });
}

void [[clang::annotate_type("webkit.nodelete")]] deleteInLambda2(SomeObject* someObj) {
  noopWithLambda([&]() {
    // expected-warning@-1{{A function 'deleteInLambda2' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    delete someObj;
  });
}

void [[clang::annotate_type("webkit.nodelete")]] deleteIfNeeded(SomeObject* someObj) {
  if (someObj)
    delete someObj;
    // expected-warning@-1{{A function 'deleteIfNeeded' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

struct MemberAssignment {
public:
  void [[clang::annotate_type("webkit.nodelete")]] clearMember() {
    m_someObject = nullptr;
    // expected-warning@-1{{A function 'clearMember' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] assignMember(SomeObject* ptr) {
    m_someObject = ptr;
    // expected-warning@-1{{A function 'assignMember' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }
  
  RefPtr<SomeObject> [[clang::annotate_type("webkit.nodelete")]] takeMember() {
    return std::exchange(m_someObject, nullptr);
    // expected-warning@-1{{A function 'takeMember' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] takeAsTemp() {
    takeMember();
    // expected-warning@-1{{A function 'takeAsTemp' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

  void [[clang::annotate_type("webkit.nodelete")]] takeToLocalVar() {
    if (RefPtr ptr = std::exchange(m_someObject, nullptr))
      // expected-warning@-1{{A function 'takeToLocalVar' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
      ptr->doTrivialWork();
  }

  void [[clang::annotate_type("webkit.nodelete")]] takeObjectsAsTemp() {
    std::exchange(m_objects, { });
    // expected-warning@-1{{A function 'takeObjectsAsTemp' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
  }

private:
  RefPtr<SomeObject> m_someObject;
  Vector<Ref<SomeObject>> m_objects;
};

namespace copy_elision_edge_cases {

// These cases all inhibit NRVO/copy elision (so a real move or copy constructor runs into the return slot),
// but none of them perform any local destruction:
//   - Moved-from operands are emptied; their dtors are no-ops at the caller.
//   - Globals/statics are not destructed here at all.
//   - The return-slot temporary is destructed by the caller, not by us.
// All the mock Ref<T> copy/move ctors only manipulate pointers and a
// refcount, so the trivial-ctor analysis correctly classifies these as
// safe. The tests below document that the checker accepts each shape.

Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnStdMoved(Ref<RefCountable>&& obj) {
  return std::move(obj);
}

Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnFromBranches(bool b, Ref<RefCountable>&& a, Ref<RefCountable>&& c) {
  if (b)
    return std::move(a);
  return std::move(c);
}

Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnDerefedParam(Ref<RefCountable>* param) {
  return *param;
}

// Returning a by-value parameter also requires a real copy/move construction.
// The function is still flagged here because of unsafe-parameter diagnostic that fires for the parameter declaration.
Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnByValueParam(Ref<RefCountable> param) {
  // expected-warning@-1{{A function 'returnByValueParam' has [[clang::annotate_type("webkit.nodelete")]] but it contains a parameter 'param' which could destruct an object}}
  return param;
}

extern Ref<RefCountable> g_ref;
Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnGlobal() {
  return g_ref;
}

struct StaticHolder {
  static Ref<RefCountable> s_ref;
};
Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnClassStatic() {
  return StaticHolder::s_ref;
}

} // namespace copy_elision_edge_cases

namespace temp_object_typecheck {

struct Tracked {
  Tracked();
  ~Tracked();
};

Tracked [[clang::annotate_type("webkit.nodelete")]] makeTracked();

struct Box {
  Box(const Tracked&) {}
};

Box [[clang::annotate_type("webkit.nodelete")]] makeBox() {
  return Box(makeTracked());
  // expected-warning@-1{{A function 'makeBox' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

} // namespace temp_object_typecheck

namespace argument_temporaries_are_not_elided {

// Only a *returned* prvalue is elided into the caller's return slot. A
// smart-pointer temporary passed as a call argument is destructed in this
// function at the end of the full-expression (the caller destroys arguments),
// so its destructor -- which may run delete -- is correctly flagged, no matter
// how the callee binds it (by value, by rvalue reference, or by const
// reference). The factory and sinks are annotated no-delete so the only
// possible offender is the argument temporary's destruction.

Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] makeRef();
void [[clang::annotate_type("webkit.nodelete")]] sinkByValue(Ref<RefCountable>);
void [[clang::annotate_type("webkit.nodelete")]] sinkByRvalueRef(Ref<RefCountable>&&);
void [[clang::annotate_type("webkit.nodelete")]] observeByConstRef(const Ref<RefCountable>&);

// Returned prvalue: constructed into the caller's return slot -> no local
// destruction here.
Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnedPrvalueIsElided() {
  return makeRef();
}

void [[clang::annotate_type("webkit.nodelete")]] passedByValueIsFlagged() {
  sinkByValue(makeRef());
  // expected-warning@-1{{A function 'passedByValueIsFlagged' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] passedByRvalueRefIsFlagged() {
  sinkByRvalueRef(makeRef());
  // expected-warning@-1{{A function 'passedByRvalueRefIsFlagged' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] passedByConstRefIsFlagged() {
  observeByConstRef(makeRef());
  // expected-warning@-1{{A function 'passedByConstRefIsFlagged' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

void [[clang::annotate_type("webkit.nodelete")]] discardedTemporaryIsFlagged() {
  makeRef();
  // expected-warning@-1{{A function 'discardedTemporaryIsFlagged' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
}

} // namespace argument_temporaries_are_not_elided

namespace returned_prvalue_typedef {

// A returned prvalue spelled through a typedef/alias is still the return-slot
// object and must be elided. The elision relies on a canonical, unqualified
// type comparison rather than exact QualType identity.

using RefRC = Ref<RefCountable>;
RefRC [[clang::annotate_type("webkit.nodelete")]] makeAlias();

Ref<RefCountable> [[clang::annotate_type("webkit.nodelete")]] returnTypedefPrvalue() {
  return makeAlias(); // no warning: the elided temporary is the return slot.
}

} // namespace returned_prvalue_typedef

namespace create_with_default_constructor {

  struct ObjectWithDefaultConstructorWithoutMemberVariables {
    void ref() const;
    void deref() const;

    static auto [[clang::annotate_type("webkit.nodelete")]] create() {
      return adoptRef(*new ObjectWithDefaultConstructorWithoutMemberVariables());
    }
  };

  struct ObjectWithDefaultConstructorWithPODMemberVariables {
    void ref() const;
    void deref() const;

    static auto [[clang::annotate_type("webkit.nodelete")]] create() {
      return adoptRef(*new ObjectWithDefaultConstructorWithPODMemberVariables());
    }

  private:
    int value { 0 };
    RefCountable* ptr { nullptr };
  };

  struct ObjectWithOpaqueCtor {
    ObjectWithOpaqueCtor();
  };

  struct ObjectWithDefaultConstructorWithOpaqueCtorMemberVariables {
    void ref() const;
    void deref() const;

    static auto [[clang::annotate_type("webkit.nodelete")]] create() {
      return adoptRef(*new ObjectWithDefaultConstructorWithOpaqueCtorMemberVariables());
      // expected-warning@-1{{A function 'create' has [[clang::annotate_type("webkit.nodelete")]] but it contains code that could destruct an object}}
    }

  private:
    ObjectWithOpaqueCtor obj;
  };

} // namespace create_with_default_constructor

struct Clazzzz {
    void ref() const;
    void deref() const;
};

Ref<Clazzzz> [[clang::annotate_type("webkit.nodelete")]] create() {
    return adoptRef(*new Clazzzz());
}

namespace trivial_implicit_ctor_in_new_expr {

// 'new T()' with parens emits a CXXConstructExpr for T's implicit default
// ctor. That ctor has no body in the AST (the synthesized body is materialised
// only at codegen), but it is trivial by the C++ standard and runs no user
// code, so it cannot delete. Verify the fast-path treats it as trivial.
struct Plain { int x; };

void [[clang::annotate_type("webkit.nodelete")]] valueInitNew() {
  Plain* p = new Plain();
  (void)p;
}

} // namespace trivial_implicit_ctor_in_new_expr
