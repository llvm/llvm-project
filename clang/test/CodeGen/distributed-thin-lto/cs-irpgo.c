// RUN: mkdir -p %t
// RUN: %clang -target x86_64-apple-darwin -flto=thin %s -fcs-profile-generate -c -o %t/main.bc
// RUN: llvm-lto2 run --thinlto-distributed-indexes %t/main.bc -r=%t/main.bc,_main,px -o %t/index
// RUN: %clang -target x86_64-apple-darwin -fcs-profile-generate -fthinlto-index=%t/main.bc.thinlto.bc %t/main.bc -c -o main.o

int main() {
    return 0;
}
