// REQUIRES: ld.lld
/// Regression test for https://github.com/llvm/llvm-project/pull/213410:
/// Check that module-level inline assembly (including .symver imported by
/// ThinLTO) and function-level inline assembly link cleanly under RegularLTO
/// and ThinLTO when targeting riscv64 with lp64d ABI and -march=rv64gcv.
// RUN: rm -rf %t && split-file %s %t
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto -c %t/a.c -o %t1.o
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto -c %t/b.c -o %t2.o
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto -shared -nostdlib -fuse-ld=lld -Wl,-save-temps -Wl,--version-script=%t/ver.ver %t1.o %t2.o -o %t.so 2>&1 \
// RUN:   | FileCheck %s --allow-empty --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
// RUN: llvm-dis %t.so.0.5.precodegen.bc -o - | FileCheck %s --check-prefix=REGULAR-IR
// RUN: llvm-readobj --file-headers %t.so | FileCheck %s --check-prefix=FLAGS
// RUN: llvm-objdump -d --show-all-symbols --no-show-raw-insn %t.so | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -t %t.so | FileCheck %s --check-prefix=SYMS --implicit-check-not='\$x'
//
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto=thin -c %t/a.c -o %t1.thin.o
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto=thin -c %t/b.c -o %t2.thin.o
// RUN: %clang --target=riscv64-linux-android -march=rv64gcv -O2 -flto=thin -shared -nostdlib -fuse-ld=lld -Wl,-save-temps -Wl,--version-script=%t/ver.ver %t1.thin.o %t2.thin.o -o %t.thin.so 2>&1 \
// RUN:   | FileCheck %s --allow-empty --implicit-check-not="error:" --implicit-check-not="warning:" --implicit-check-not="note:"
// RUN: llvm-dis %t2.thin.o.5.precodegen.bc -o - | FileCheck %s --check-prefix=THIN-IR
// RUN: llvm-readobj --file-headers %t.thin.so | FileCheck %s --check-prefix=FLAGS
// RUN: llvm-objdump -d --show-all-symbols --no-show-raw-insn %t.thin.so | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -t %t.thin.so | FileCheck %s --check-prefix=SYMS --implicit-check-not='\$x'
//
// REGULAR-IR:      module asm(target_features: "+64bit,{{.*}}", target_cpu: "generic-rv64")
// REGULAR-IR-NEXT:     ".lto_discard "
// REGULAR-IR-NEXT:     "nop"
// REGULAR-IR-NEXT:     ".symver symver_fn, symver_fn@VER_1.0"
//
// THIN-IR:      module asm(target_features: "+64bit,{{.*}}", target_cpu: "generic-rv64")
// THIN-IR-NEXT:     ".symver symver_fn, symver_fn@VER_1.0"
//
// FLAGS:      Flags [ (0x5)
// FLAGS-NEXT:   EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
// FLAGS-NEXT:   EF_RISCV_RVC (0x1)
// FLAGS-NEXT: ]
//
/// TODO: RISCVTargetELFStreamer::emitTextAttribute does not update the
/// streamer's ArchString when emitting the module's RISCVAttrs::ARCH attribute
/// ("rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_...").
// DISASM-LABEL: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT:  [[#%x,]] <$xrv64i2p1>:
// DISASM-NEXT:  [[#%x,]]:      	nop
// DISASM-EMPTY:
// DISASM-NEXT:  [[#%x,]] <$xrv64i2p1>:
// DISASM-NEXT:  [[#%x,]] <symver_fn>:
// DISASM-NEXT:  [[#%x,]]:      	ret
// DISASM-EMPTY:
// DISASM-NEXT:  [[#%x,]] <$xrv64i2p1>:
// DISASM-NEXT:  [[#%x,]] <fn>:
// DISASM-NEXT:  [[#%x,]]:      	nop
// DISASM-NEXT:  [[#%x,]]:      	ret
// DISASM-EMPTY:
// DISASM-NEXT:  [[#%x,]] <$xrv64i2p1>:
// DISASM-NEXT:  [[#%x,]] <caller>:
// DISASM-NEXT:  [[#%x,]]:      	nop
// DISASM-NEXT:  [[#%x,]]:      	ret
// DISASM-NOT:   {{.}}
//
// SYMS: [[#%x,]] l       .text	0000000000000000 $xrv64i2p1{{$}}
// SYMS: [[#%x,]] l       .text	0000000000000000 $xrv64i2p1{{$}}
// SYMS: [[#%x,]] l       .text	0000000000000000 $xrv64i2p1{{$}}
// SYMS: [[#%x,]] l       .text	0000000000000000 $xrv64i2p1{{$}}
// SYMS: [[#%x,]] g     F .text	0000000000000004 fn{{$}}
// SYMS: [[#%x,]] g     F .text	0000000000000002 symver_fn{{$}}
// SYMS: [[#%x,]] g     F .text	0000000000000004 caller{{$}}

//--- ver.ver
VER_1.0 {};

//--- a.c
__asm__("nop");
__asm__(".symver symver_fn, symver_fn@VER_1.0");

void symver_fn(void) {}

void fn(void) { __asm__ volatile("nop"); }

//--- b.c
extern void fn(void);
extern void symver_fn(void);
void caller(void) {
  fn();
  symver_fn();
}
