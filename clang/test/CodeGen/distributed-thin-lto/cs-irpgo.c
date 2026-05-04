// RUN: mkdir -p %t
// RUN: %clang -target x86_64-apple-darwin -fcs-profile-generate -flto=thin -c -o %t/main.bc %s
// RUN: llvm-lto2 run --thinlto-distributed-indexes %t/main.bc -o %t/index -r=%t/main.bc,_main,px
// RUN: %clang -target x86_64-apple-darwin -fcs-profile-generate -c -o %t/main.o -fthinlto-index=%t/main.bc.thinlto.bc %t/main.bc

int main() {
    return 0;
}
