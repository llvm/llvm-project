// RUN: %clang -g -target bpf -S -emit-llvm %s -o - | FileCheck %s
//
// When linking BPF object files via bpftool, BTF info is required for
// every symbol. BTF is generated from debug info. Ensure that debug info
// is emitted for extern functions referenced via variable initializers.
//
// CHECK: !DISubprogram(name: "fn"
extern void fn(void);
void (*pfn) (void) = &fn;
