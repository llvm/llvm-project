// RUN: %clang_cc1 -emit-llvm %s -o /dev/null -debug-info-kind=limited

template <typename T1,typename T2>
inline void f(const T1&,const T2&) { }

template <typename T1,typename T2,void F(const T1&,const T2&)>
struct A {
    template <typename T> void g(T& i) { }
};

int main() {
    int i;
    A<int,int,f> a;
    a.g(i);
}
