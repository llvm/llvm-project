// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncheckedCallArgsChecker -verify %s

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
  // expected-warning@-1{{Call argument for 'this' parameter is unchecked and unsafe}}
}

void doWorkWithObject(const CheckedObject&);
void bar() {
  doWorkWithObject(CheckedObject());
}
