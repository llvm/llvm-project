// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -emit-llvm -o - %s | FileCheck %s

// The union's IR type is struct work, so num_layers has to be reached by byte
// offset rather than through struct work's field indices.

struct list {
  struct list *next, *prev;
};

struct work {
  long data;
  struct list entry;
  void (*func)(void);
};

struct domain {
  void *rules[3];
  void *hierarchy;
  union {
    struct work work_free;
    struct {
      int usage;
      unsigned int num_layers;
      unsigned int masks[] __attribute__((counted_by(num_layers)));
    };
  };
};

// CHECK-LABEL: define {{.*}} @test_bdos(
// CHECK: [[GEP:%.*]] = getelementptr inbounds nuw i8, ptr %d, i64 36
// CHECK: load i32, ptr [[GEP]]
unsigned long test_bdos(struct domain *d) {
  return __builtin_dynamic_object_size(d->masks, 1);
}

// A bit-field count can't be loaded by byte offset, so the size is unknown.
struct bitfield_count {
  unsigned int pad : 20;
  unsigned int count : 12;
  int fam[] __attribute__((counted_by(count)));
};

// CHECK-LABEL: define {{.*}} @test_bitfield_count(
// CHECK: ret i64 -1
unsigned long test_bitfield_count(struct bitfield_count *p) {
  return __builtin_dynamic_object_size(p->fam, 1);
}
