// clang bytes1.cl --target=amdgcn-amdhsa-opencl -mcpu=gfx803 -c -o bytes1.o

void kernel foo(global int *a) { *a = 42; }
