inline char *not_exempt_cast(int *p) {
  return reinterpret_cast<char *>(p);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
}
