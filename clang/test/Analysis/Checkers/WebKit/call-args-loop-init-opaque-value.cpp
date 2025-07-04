// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedCallArgsChecker -verify %s

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

struct NonTrivial {
  NonTrivial();
  NonTrivial(const NonTrivial&);
  bool operator==(const NonTrivial& other) const { return value == other.value; }
  float value;
};

class Component {
public:
    void ref() const;
    void deref() const;

    Obj<float, 4> unresolvedComponents() const { return m_components; }
    Obj<NonTrivial, 4> unresolvedNonTrivialComponents() const { return m_nonTrivialComponents; }

    bool isEqual(const Component& other) const {
        return unresolvedComponents() == other.unresolvedComponents();
    }

    bool isNonTrivialEqual(const Component& other) const {
        return unresolvedNonTrivialComponents() == other.unresolvedNonTrivialComponents();
    }

private:
    Obj<float, 4> m_components;
    Obj<NonTrivial, 4> m_nonTrivialComponents;
};

Component* provide();
bool someFunction(Component* other) {
    return provide()->isEqual(*other);
}

bool otherFunction(Component* other) {
    return provide()->isNonTrivialEqual(*other);
    // expected-warning@-1{{Call argument for 'this' parameter is uncounted and unsafe}}
}
