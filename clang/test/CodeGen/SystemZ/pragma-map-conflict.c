// RUN: %clang_cc1 %s -triple s390x-ibm-zos -emit-llvm -O1 -o - | FileCheck %s
// Give a warning and ignore the pragma map.
void rp(void) asm("rp_ext");
void rp(void);
#pragma map(rp, "a187")

void rp_ext(void);


void rp_orig(void) asm("a187");

void rp_ext(void) {
  rp_orig();
}
//CHECK: define void @rp_ext() local_unnamed_addr #0 {
//CHECK:     call void @a187()
//CHECK: }

//CHECK: declare void @a187()
