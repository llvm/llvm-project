// RUN: %clang_cc1 -fsyntax-only -verify=enabled -Wcounted-by-addrof %s
// RUN: %clang_cc1 -fsyntax-only -verify=disabled %s
// RUN: %clang_cc1 -fsyntax-only -Wcounted-by-addrof -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// disabled-no-diagnostics

#define __counted_by(f) __attribute__((counted_by(f)))

typedef __SIZE_TYPE__ size_t;

struct annotated_flex {
  size_t count;
  char induce_padding;
  char fam[] __counted_by(count);
};

struct plain_flex {
  size_t count;
  char fam[];
};

struct annotated_flex *get_ptr(void);

size_t ptr_addrof(struct annotated_flex *p) {
  return __builtin_dynamic_object_size(&p->fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

// Opaque pointer return. Allocation not statically known. fix-it offered.
size_t ret_addrof(void) {
  return __builtin_dynamic_object_size(&get_ptr()->fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

// Subscripting a pointer escapes to an unknown allocation. fix-it offered.
size_t ptr_subscript_addrof(struct annotated_flex *parr, int i) {
  return __builtin_dynamic_object_size(&parr[i].fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

struct annotated_flex gaf;

// Global (static storage). No fix-it.
size_t global_addrof(void) {
  return __builtin_dynamic_object_size(&gaf.fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

// Local (automatic storage). No fix-it.
size_t local_addrof(size_t n) {
  struct annotated_flex af;
  af.count = n;
  return __builtin_dynamic_object_size(&af.fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

// Static local (static storage, local scope). No fix-it.
size_t static_local_addrof(void) {
  static struct annotated_flex saf;
  return __builtin_dynamic_object_size(&saf.fam, 1); // enabled-warning {{taking the address of flexible array member 'fam' discards the 'counted_by' bound}}
}

size_t decayed(struct annotated_flex *p) {
  // Decayed pointer-to-element honors the count. There is no '&'.
  return __builtin_dynamic_object_size(p->fam, 1);
}

size_t element_addrof(struct annotated_flex *p, int i) {
  // Address of an element, not the FAM-as-a-whole.
  return __builtin_dynamic_object_size(&p->fam[i], 1);
}

char *count_addrof(struct annotated_flex *p) {
  return (char *)&p->count; // non-FAM field.
}

char (*plain_addrof(struct plain_flex *p))[] {
  return &p->fam; // FAM without counted_by.
}

// CHECK-COUNT-3: fix-it:{{.*}}:""
// CHECK-NOT: fix-it: