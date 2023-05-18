struct X { union { int n; }; };
inline int a(X x) { return x.n; }
