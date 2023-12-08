// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

// RUN: %clang_cc1 -std=c++14 -x c++-header %s -emit-pch -fpch-instantiate-templates -o %t.pch
// RUN: %clang_cc1 -std=c++14 -x c++ /dev/null -include-pch %t.pch

template <class T, T... I>
struct Seq {
    static constexpr T PackSize = sizeof...(I);
};

template <typename T, T N>
using MakeSeq = __make_integer_seq<Seq, T, N>;

void fn1() {
  MakeSeq<int, 3> x;
}
