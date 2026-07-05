// RUN: %clang_cc1 %s -emit-llvm -o -
// PR906

struct state_struct {
  unsigned long long phys_frame: 50;
  unsigned valid : 2;
} s;

int mem_access(struct state_struct *p) {
  return p->valid;
}

