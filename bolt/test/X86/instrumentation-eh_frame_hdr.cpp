// This test checks that .eh_frame_hdr address is in bounds of the last LOAD
// end address i.e. the section address is smaller then the LOAD end address.

// REQUIRES: system-linux,bolt-runtime,target=x86_64{{.*}}

// RUN: %clangxx %cxxflags -static -Wl,-q %s -o %t.exe -Wl,--entry=_start
// RUN: llvm-bolt %t.exe -o %t.instr -instrument \
// RUN:   --instrumentation-file=%t.fdata -instrumentation-sleep-time=1
// RUN: llvm-readelf -SW %t.instr | grep -v bolt > %t.sections
// RUN: llvm-readelf -lW %t.instr | grep LOAD | tail -n 1 >> %t.sections
// RUN: FileCheck %s < %t.sections

// CHECK: {{.*}} .eh_frame_hdr PROGBITS [[#%x, EH_ADDR:]]
// CHECK: LOAD 0x[[#%x, LD_OFFSET:]] 0x[[#%x, LD_VADDR:]] 0x[[#%x, LD_FSIZE:]]
// CHECK-SAME: 0x[[#%x, LD_MEMSIZE:]]
//
// If .eh_frame_hdr address bigger then last LOAD segment end address test would
// fail with overflow error, otherwise the result of the expression is 0 that
// could be found on this line e.g. in LOAD align field.
// CHECK-SAME: [[#LD_VADDR + LD_MEMSIZE - max(LD_VADDR + LD_MEMSIZE,EH_ADDR)]]

#include <cstdio>
#include <stdexcept>

void foo() { throw std::runtime_error("Exception from foo()"); }

void bar() { foo(); }

int main() {
  try {
    bar();
  } catch (const std::exception &e) {
    printf("Exception caught: %s\n", e.what());
  }
}

extern "C" {
void _start();
}

void _start() { main(); }
