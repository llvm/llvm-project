// REQUIRES: x86-registered-target
// RUN: %clang -c -Wno-tentative-definition-array -Wno-return-type -Wno-unused-value -Wno-array-bounds -Xclang -verify --target=x86_64-- -fsanitize=undefined %s 
int **a[];
int main() {
  (*a)[3300220222222200000]; // expected-error {{Expression caused pointer calculation overflow during code generation}}
  return 0;
}
