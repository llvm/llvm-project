// RUN: %clang_cc1 -emit-llvm -dwarf-version=2 -debug-info-kind=limited -fblocks -o - -x objective-c %s| FileCheck %s
// This code triggered a bug where a dbg.declare intrinsic ended up with the
// wrong parent and subsequently failed the Verifier.
void baz(id b);
void fub(id block);
int foo(void);
void bar(void) {
  fub(^() {
      id a;
      id b = [a bar:^(int e){}];
      if (b) {
        ^() {
            if ((0 && foo()) ? 1 : 0) {
              baz([a aMessage]);
            }
        };
      }
  });
}

// Verify that debug info for BlockPointerDbgLoc is emitted for the
// innermost block.
//
// CHECK: define {{.*}}void @__bar_block_invoke_3(ptr noundef %.block_descriptor)
// CHECK: %[[BLOCKADDR:.*]] = alloca ptr, align
// CHECK: #dbg_declare({{.*}}%[[BLOCKADDR]]
