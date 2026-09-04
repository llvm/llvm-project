// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

#include "mock-types.h"

class Object {
public:
    void ref() const;
    void deref() const;

    bool constFunc() const;
    void mutableFunc();
};

class Caller {
  void someFunction();
  void otherFunction();
private:
    RefPtr<Object> m_obj;
};

void Caller::someFunction()
{
    m_obj->constFunc();
    // expected-warning@-1{{Function argument 'this->m_obj' (parameter 'this' to 'Object::constFunc') is a raw pointer to RefPtr-capable type 'Object'}}
    m_obj->mutableFunc();
    // expected-warning@-1{{Function argument 'this->m_obj' (parameter 'this' to 'Object::mutableFunc') is a raw pointer to RefPtr-capable type 'Object'}}
}
