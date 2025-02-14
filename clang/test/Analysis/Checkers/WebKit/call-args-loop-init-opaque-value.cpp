// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s
// expected-no-diagnostics

typedef unsigned long size_t;
template<typename T, size_t N>
struct Obj {
    constexpr static size_t Size = N;

    constexpr T& operator[](size_t i) { return components[i]; }
    constexpr const T& operator[](size_t i) const { return components[i]; }

    constexpr size_t size() const { return Size; }

    T components[N];
};

template<typename T, size_t N>
constexpr bool operator==(const Obj<T, N>& a, const Obj<T, N>& b)
{
    for (size_t i = 0; i < N; ++i) {
        if (a[i] == b[i])
            continue;
        return false;
    }

    return true;
}

class Component {
public:
    void ref() const;
    void deref() const;

    Obj<float, 4> unresolvedComponents() const { return m_components; }

    bool isEqual(const Component& other) const {
        return unresolvedComponents() == other.unresolvedComponents();
    }

private:
    Obj<float, 4> m_components;
};

Component* provide();
bool someFunction(Component* other) {
    return provide()->isEqual(*other);
}
