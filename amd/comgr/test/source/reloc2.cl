// clang bytes2.cl --target=amdgcn-amdhsa-opencl -mcpu=gfx900 -c -o bytes2.o

void kernel bar(global int *a) { *a = 43; }
