//===-- test.c ------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Sanity test for Go runtime.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

void __tsan_init(void **thr, void (*cb)(void*));
void __tsan_fini();
void __tsan_map_shadow(void *addr, unsigned long size);
void __tsan_go_start(void *thr, void **chthr, void *pc);
void __tsan_go_end(void *thr);
void __tsan_read(void *thr, void *addr, void *pc);
void __tsan_write(void *thr, void *addr, void *pc);
void __tsan_func_enter(void *thr, void *pc);
void __tsan_func_exit(void *thr);
void __tsan_malloc(void *p, unsigned long sz);
void __tsan_acquire(void *thr, void *addr);
void __tsan_release(void *thr, void *addr);
void __tsan_release_merge(void *thr, void *addr);

void symbolize_cb(void *ctx) {}

char buf0[100<<10];

void foobar() {}
void barfoo() {}

int main(void) {
  void *thr0 = 0;
  char *buf = (char*)((unsigned long)buf0 + (64<<10) - 1 & ~((64<<10) - 1));
  __tsan_malloc(buf, 10);
  __tsan_init(&thr0, symbolize_cb);
  __tsan_map_shadow(buf, 4096);
  __tsan_func_enter(thr0, (char*)&main + 1);
  __tsan_malloc(buf, 10);
  __tsan_release(thr0, buf);
  __tsan_release_merge(thr0, buf);
  void *thr1 = 0;
  __tsan_go_start(thr0, &thr1, (char*)&barfoo + 1);
  void *thr2 = 0;
  __tsan_go_start(thr0, &thr2, (char*)&barfoo + 1);
  __tsan_func_enter(thr1, (char*)&foobar + 1);
  __tsan_func_enter(thr1, (char*)&foobar + 1);
  __tsan_write(thr1, buf, (char*)&barfoo + 1);
  __tsan_acquire(thr1, buf);
  __tsan_func_exit(thr1);
  __tsan_func_exit(thr1);
  __tsan_go_end(thr1);
  __tsan_func_enter(thr2, (char*)&foobar + 1);
  __tsan_read(thr2, buf, (char*)&barfoo + 1);
  __tsan_func_exit(thr2);
  __tsan_go_end(thr2);
  __tsan_func_exit(thr0);
  __tsan_fini();
  return 0;
}
