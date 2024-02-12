// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

#include "mock-types.h"

class Base {
public:
    virtual ~Base();
    void ref() const;
    void deref() const;
};

class Event : public Base {
protected:
    explicit Event();
};

class SubEvent : public Event {
public:
    static Ref<SubEvent> create();
private:
    SubEvent() = default;
};

void someFunction(Base&);

static void test()
{
    someFunction(SubEvent::create());
}
