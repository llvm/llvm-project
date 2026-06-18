
template <typename F> struct A{};

template <typename Ret, typename C, typename... Args> struct A<Ret (             C::*)(Args...) noexcept> { static constexpr int value = 0; };
template <typename Ret, typename C, typename... Args> struct A<Ret (__vectorcall C::*)(Args...) noexcept> { static constexpr int value = 1; };

template <typename F> constexpr int A_v = A<F>::value;

struct B
{
    void f() noexcept {}
    void __vectorcall g() noexcept {}
};

int main()
{
    return A_v<decltype(&B::f)> + A_v<decltype(&B::g)>;
}
