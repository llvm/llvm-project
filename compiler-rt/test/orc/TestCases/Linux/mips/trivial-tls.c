// RUN: %clang -O1 -fPIC -fno-stack-protector -c -o %t %s
// RUN: %llvm_jitlink %t
// RUN: %clang -O1 -fPIC -fno-stack-protector -S -emit-llvm -o %t.ll %s
// RUN: %lli_orc_jitlink -emulated-tls=false -relocation-model=pic %t.ll

// Check general-dynamic and local-dynamic TLS in .tdata and .tbss.

__thread __attribute__((tls_model("global-dynamic"))) int gd_data = 7;
__thread __attribute__((tls_model("global-dynamic"))) int gd_bss;
static __thread
    __attribute__((tls_model("local-dynamic"))) volatile int ld_data = 11;
static __thread __attribute__((tls_model("local-dynamic"))) volatile int ld_bss;

int main(void) { return gd_data + gd_bss + ld_data + ld_bss - 18; }
