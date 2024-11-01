struct S { int i, j; };

constexpr S foo(int i, int j) { return S(i, j); };

void bar(int i, int j) { int arr[4](i, j); };
