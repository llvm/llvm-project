// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' --cflags "-O0 -g" \
// RUN:     --ldflags="-lstdc++" -- %s
// Radar 8945514

class SVal {
public:
  ~SVal() {}
  const void* Data;
  unsigned Kind;
};

void bar(SVal &v) {}
class A {
public:
  void foo(SVal v) { bar(v); } // DexLabel('foo')
};

int main() {
  SVal v;
  v.Data = 0;
  v.Kind = 2142;
  A a;
  a.foo(v);
  return 0;
}

/*
DexExpectProgramState({
  'frames': [
    {
      'location': { 'lineno': 'foo' },
      'watches': {
        'v.Data == 0': 'true',
        'v.Kind': '2142'
      }
    }
  ]
})
*/

