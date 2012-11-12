// RUN: %clang -emit-llvm -S -O0 -g %s -o - | grep DW_TAG_typedef | grep float4
typedef float float4 __attribute__((vector_size(16)));

int main(){
  volatile float4 x = (float4) { 0.0f, 1.0f, 2.0f, 3.0f };
  x += x;
  return 0;
}

