struct S { int i, j; };

union U { unsigned : 8; int i; char j; };

constexpr S foo(int i, int j) { return S(i, j); }

void bar(int i, int j) { int arr[4](i, j); }

constexpr U baz(int i) { return U(i); }
