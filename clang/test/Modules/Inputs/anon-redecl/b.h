struct X { union { int n; }; };
inline int b(X x) { return x.n; }
