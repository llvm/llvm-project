// REQUIRES: plugins, llvm-examples

// Entry-points in default and -O0 pipeline
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP %s
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -flto=full -O0 \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP %s
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -flto=thin -O0 \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP %s
//
// EP:     PipelineStart
// EP:     PipelineEarlySimplification
// EP-NOT: Peephole
// EP-NOT: ScalarOptimizerLate
// EP-NOT: Peephole
// EP:     OptimizerEarly
// EP-NOT: Vectorizer
// EP:     OptimizerLast

// Entry-points in optimizer pipeline
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP-OPT %s
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -flto=full \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP-OPT %s
//
// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -ffat-lto-objects \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP-OPT %s
//
// EP-OPT: PipelineStart
// EP-OPT: PipelineEarlySimplification
// EP-OPT: Peephole
// EP-OPT: ScalarOptimizerLate
// EP-OPT: Peephole
// EP-OPT: OptimizerEarly
// EP-OPT: VectorizerStart
// EP-OPT: VectorizerEnd
// EP-OPT: OptimizerLast

// RUN: %clang -fpass-plugin=%llvmshlibdir/Bye%pluginext -O2 -flto=thin \
// RUN:        -Xclang -load -Xclang %llvmshlibdir/Bye%pluginext \
// RUN:        -mllvm -print-ep-callbacks -o /dev/null -S -emit-llvm %s | FileCheck --check-prefix=EP-LTO-THIN %s
//
// EP-LTO-THIN:     PipelineStart
// EP-LTO-THIN:     PipelineEarlySimplification
// EP-LTO-THIN:     Peephole
// EP-LTO-THIN:     ScalarOptimizerLate
// EP-LTO-THIN:     OptimizerEarly
// EP-LTO-THIN-NOT: Vectorizer
// EP-LTO-THIN:     OptimizerLast

int f(int x) {
  return x;
}
