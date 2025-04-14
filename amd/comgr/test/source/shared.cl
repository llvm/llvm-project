// Standard
// clang shared.cl --target=amdgcn-amd-amdhsa -mcpu=gfx900 -O3 -o shared.so

__attribute__((visibility("default"))) constant int foo = 0;

void kernel bazzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz(
    global int *a, const global int *b) {
  *a = *b;
}
