#pragma GCC system_header

inline int system_header_func(int x)
  __attribute__((diagnose_if(x == x, "system header warning", "warning"))) // expected-note {{from 'diagnose_if' attribute}}
{
  return 0;
}

void test_system_header() {
  system_header_func(0); // expected-warning {{system header warning}}
}
