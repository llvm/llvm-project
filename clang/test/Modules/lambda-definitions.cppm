// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/invocable.cppm -emit-module-interface -o %t/invocable.pcm
// RUN: %clang_cc1 -std=c++20 %t/lambda.cppm -emit-module-interface -o %t/lambda.pcm  -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/invocable.cppm -emit-reduced-module-interface -o %t/invocable.pcm
// RUN: %clang_cc1 -std=c++20 %t/lambda.cppm -emit-reduced-module-interface -o %t/lambda.pcm  -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/test.cc -fprebuilt-module-path=%t -fsyntax-only -verify

//--- invocable.cppm
export module invocable;
export template <class _Fn, class... _Args>
concept invocable = requires(_Fn&& __fn, _Args&&... __args) {
  _Fn(__args...);
};

export template <class _Fn, class _Args>
constexpr bool is_callable(_Fn&& __fn, _Args&& __args) {
    return invocable<_Fn, _Args>;
}

export template <class _Fn>
struct Callable : _Fn {
    constexpr explicit Callable(_Fn &&__fn) : _Fn(static_cast<_Fn&&>(__fn)) {}
    
    template <class _Args>
    constexpr auto operator()(_Args&& __args) {
        return _Fn(__args);
    }
};

//--- lambda.cppm
export module lambda;
import invocable;
export constexpr auto l = Callable([](auto &&x){});

//--- test.cc
// expected-no-diagnostics
import invocable;
import lambda;

static_assert(is_callable(l, 4) == true);
