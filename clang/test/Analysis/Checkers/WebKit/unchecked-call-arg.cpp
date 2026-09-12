// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

#include "mock-types.h"

void WTFCrash(void);

enum class Tag : bool { Value };

template <typename StorageType, Tag> class CanMakeCheckedPtrBase {
public:
  void incrementCheckedPtrCount() const { ++m_checkedPtrCount; }
  inline void decrementCheckedPtrCount() const
  {
      if (!m_checkedPtrCount)
        WTFCrash();
      --m_checkedPtrCount;
  }

private:
  mutable StorageType m_checkedPtrCount { 0 };
};

template<typename T, Tag tag>
class CanMakeCheckedPtr : public CanMakeCheckedPtrBase<unsigned int, tag> {
};

class CheckedObject : public CanMakeCheckedPtr<CheckedObject, Tag::Value> {
public:
  void doWork();
};

CheckedObject* provide();
void foo() {
  provide()->doWork();
  // expected-warning@-1{{Function argument 'provide()' (parameter 'this' to 'CheckedObject::doWork') is a raw pointer to CheckedPtr-capable type 'CheckedObject'}}
}

void doWorkWithObject(const CheckedObject&);
void bar() {
  doWorkWithObject(CheckedObject());
}

namespace refptr_checked_ptr_capable {

class CheckedRefCounted {
public:
  void ref() const;
  void deref() const;
  void incrementCheckedPtrCount() const;
  void decrementCheckedPtrCount() const;
};

void receive(CheckedRefCounted&);
struct Foo {
  Ref<CheckedRefCounted> m_obj;

  void foo() {
    receive(m_obj.copyRef());
  }
};

} // namespace refptr_checked_ptr_capable
