// REQUIRES: x86_64-linux, gnu_ld
//
// RUN: split-file %s %t
// RUN: %clang %cflags -fPIC -shared -o %t/libexample.so \
// RUN:   %t/example.c
// RUN: %clang %cflags -fno-pie -no-pie -fuse-ld=bfd \
// RUN:   -Wl,--emit-relocs -Wl,-rpath,\$ORIGIN -o %t/main %t/main.c \
// RUN:   -L%t -lexample
// RUN: llvm-readelf --dyn-relocations %t/main | \
// RUN:   sed -n "/'PLT' relocation section/,/^$/p" | \
// RUN:   awk '/^0/ { print $3, $5 }' > %t/main.plt
// RUN: FileCheck %s --check-prefix=PLT-KINDS < %t/main.plt
// RUN: llvm-bolt %t/main -o %t/main.bolt
// RUN: llvm-readelf --dyn-relocations %t/main.bolt | \
// RUN:   sed -n "/'PLT' relocation section/,/^$/p" | \
// RUN:   awk '/^0/ { print $3, $5 }' > %t/main.bolt.plt
// RUN: diff %t/main.plt %t/main.bolt.plt
// RUN: %t/main.bolt

// PLT-KINDS-DAG: R_X86_64_JUMP_SLOT long_name
// PLT-KINDS-DAG: R_X86_64_IRELATIVE

//--- main.c
extern int long_name(void);

__attribute__((target_clones("default,avx2"))) int foo(int x) { return x + 1; }

int main(void) {
  if (foo(1) != 2)
    return 1;
  if (long_name() != 0)
    return 1;
  return 0;
}

//--- example.c
int long_name(void) { return 0; }
