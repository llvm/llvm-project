// Debug
// clang -c -O3 -g -target=amdgcn-amd-amdhsa -mcpu=gfx900 symbolize.cl -o
// symbolize-debug.so

__attribute__((visibility("default"))) constant int foo = 1234;

int offset(int x) { return x + foo + 5678; }

void kernel bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz(
    global int *a, const global int *b) {
  if (offset(foo) < offset(*b))
    *a = *b;
  else
    *a = foo;
}
