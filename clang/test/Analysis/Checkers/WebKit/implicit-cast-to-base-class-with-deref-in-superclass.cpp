// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

template<typename T>
class Ref {
public:
    ~Ref()
    {
        if (auto* ptr = m_ptr)
            ptr->deref();
        m_ptr = nullptr;
    }

    Ref(T& object)
        : m_ptr(&object)
    {
        object.ref();
    }

    operator T&() const { return *m_ptr; }
    bool operator!() const { return !*m_ptr; }

private:
    T* m_ptr;
};

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
