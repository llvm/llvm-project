// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

template<typename T>
class RefPtr {
public:
    inline constexpr RefPtr() : m_ptr(nullptr) { }
    inline RefPtr(T* ptr)
      : m_ptr(ptr)
    {
        if (m_ptr)
            m_ptr->ref();
    }

    inline RefPtr(const RefPtr& o)
        : m_ptr(o.m_ptr)
    {
        if (m_ptr)
            m_ptr->ref();
    }

    inline ~RefPtr()
    {
        if (m_ptr)
            m_ptr->deref();
    }

    inline T* operator->() const { return m_ptr; }
    explicit operator bool() const { return m_ptr; }
  
    RefPtr& operator=(const RefPtr&);
    RefPtr& operator=(T*);

private:
    T* m_ptr;
};

template<typename T>
inline RefPtr<T>& RefPtr<T>::operator=(const RefPtr& o)
{
    if (m_ptr)
        m_ptr->deref();
    m_ptr = o.m_ptr;
    if (m_ptr)
        m_ptr->ref();
    return *this;
}

template<typename T>
inline RefPtr<T>& RefPtr<T>::operator=(T* optr)
{
    if (m_ptr)
        m_ptr->deref();
    m_ptr = optr;
    if (m_ptr)
        m_ptr->ref();
    return *this;
}

class Node {
public:
    Node* nextSibling() const;

    void ref() const;
    void deref() const;
};

static void removeDetachedChildren(Node* firstChild)
{
    for (RefPtr child = firstChild; child; child = child->nextSibling());
}
