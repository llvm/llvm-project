// RUN: %clang -O1 -fPIC -c -o %t %s
// RUN: %llvm_jitlink %t
// RUN: %clang -O1 -fPIC -emit-llvm -c -o %t.bc %s
// RUN: %lli_orc_jitlink -emulated-tls=false -relocation-model=pic %t.bc

// Check GD and LD TLS in .tdata and .tbss.
__thread int gd_data = 7;
__thread int gd_bss;
static __thread volatile int ld_data = 11;
static __thread volatile int ld_bss;

int main(void) { return gd_data + gd_bss + ld_data + ld_bss - 18; }
