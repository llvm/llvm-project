struct X { union { int n; }; };
inline int c(X x) { return x.n; }
