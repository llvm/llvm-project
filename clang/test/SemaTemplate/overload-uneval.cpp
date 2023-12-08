// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused %s
// expected-no-diagnostics

// Tests that overload resolution is treated as an unevaluated context.
// PR5541
struct Foo
{
    Foo *next;
};

template <typename>
struct Bar
{
};


template <typename T>
class Wibble
{
    typedef Bar<T> B;

    static inline B *concrete(Foo *node) {
        int a[sizeof(T) ? -1 : -1];
        return reinterpret_cast<B *>(node);
    }

public:
    class It
    {
        Foo *i;

    public:
        inline operator B *() const { return concrete(i); }
        inline bool operator!=(const It &o) const { return i !=
o.i; }
    };
};

void f() {
  Wibble<void*>::It a, b;

  a != b;
}
