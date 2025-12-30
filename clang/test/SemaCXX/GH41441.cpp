// RUN: %clang --target=x86_64-pc-linux -S -fno-discard-value-names -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 %s -fsyntax-only -verify

namespace std {
  using size_t = decltype(sizeof(int));
};
void* operator new[](std::size_t, void*) noexcept;

// CHECK: call void @llvm.memset.p0.i64(ptr align 1 %x, i8 0, i64 8, i1 false)
// CHECK: call void @llvm.memset.p0.i64(ptr align 16 %x, i8 0, i64 32, i1 false)
template <typename TYPE>
void f()
{
    typedef TYPE TArray[8];

    TArray x;
    new(&x) TArray();
}

template <typename T>
void f1() {
  int (*x)[1] = new int[1][1];
}
template void f1<char>();
void f2() {
  int (*x)[1] = new int[1][1];
}

int main()
{
    f<char>();
    f<int>();
}

// expected-no-diagnostics
template <typename T> struct unique_ptr {unique_ptr(T* p){}};

template <typename T>
unique_ptr<T> make_unique(unsigned long long n) {
  return unique_ptr<T>(new T[n]());
}

auto boro(int n){
	typedef double HistoryBuffer[4];
	return make_unique<HistoryBuffer>(n);
}
