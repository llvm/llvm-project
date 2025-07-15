// RUN: env SDKROOT="/" %clang -Wno-error=return-type -emit-llvm -S -o - -x c++ - < %s | grep -v DIFile > %t1.ll
// RUN: env SDKROOT="/" %clang -Wno-error=return-type -fno-delayed-template-parsing -emit-ast -o %t.ast %s
// RUN: env SDKROOT="/" %clang -Wno-error=return-type -emit-llvm -S -o - -x ast - < %t.ast | grep -v DIFile > %t2.ll
// RUN: diff %t1.ll %t2.ll

// http://llvm.org/bugs/show_bug.cgi?id=15377
template<typename T>
struct S {
    T *mf();
};
template<typename T>
T *S<T>::mf() {
    // warning: non-void function does not return a value [-Wreturn-type]
}

void f() {
    S<int>().mf();
}

int main() {
  return 0;
}
