// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s

template <class T>
T&&
declval() noexcept;

template <class T>
struct some_trait
{
    static const bool value = false;
};

template <class T>
void swap(T& x, T& y) noexcept(some_trait<T>::value)
{
    T tmp(static_cast<T&&>(x));
    x = static_cast<T&&>(y);
    y = static_cast<T&&>(tmp);
}

template <class T, unsigned N>
struct array
{
    T data[N];

  void swap(array& a) noexcept(noexcept(::swap(declval<T&>(), declval<T&>())));
};

struct DefaultOnly
{
    DefaultOnly() = default;
    DefaultOnly(const DefaultOnly&) = delete;
    DefaultOnly& operator=(const DefaultOnly&) = delete;
    ~DefaultOnly() = default;
};

int main()
{
    array<DefaultOnly, 1> a, b;
}

