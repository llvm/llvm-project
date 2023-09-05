// REQUIRES: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w --builder 'clang-cl_vs2015' \
// RUN:      --debugger 'dbgeng' --cflags '/Z7 /Zi' --ldflags '/Z7 /Zi' -- %s

// From https://llvm.org/pr38857, where we had issues with stack realignment.

struct Foo {
  int x = 42;
  int __declspec(noinline) foo();
  void __declspec(noinline) bar(int *a, int *b, double *c);
};
int Foo::foo() {
  int a = 1;
  int b = 2;
  double __declspec(align(32)) force_alignment = 0.42;
  bar(&a, &b, &force_alignment); // DexLabel('in_foo')
  x += (int)force_alignment;
  return x;
}
void Foo::bar(int *a, int *b, double *c) {
  *c += *a + *b; // DexLabel('in_bar')
}
int main() {
  Foo o;
  o.foo();
}
/*
DexExpectProgramState({'frames':[
    {'function': 'Foo::bar', 'location' : {'lineno' : ref('in_bar')} },
    {'function': 'Foo::foo',
     'watches' : {
       'a' : '1',
       'b' : '2',
       'force_alignment' : '0.42'
     }
    }
]})
*/
