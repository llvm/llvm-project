// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o %t

typedef struct _attrs {
        unsigned file_attributes;
        unsigned char filename_length;
} __attribute__((__packed__)) attrs;

// RUN: grep "union._attr_union = type <{ i32, i8 }>" %t
typedef union _attr_union {
  attrs file_attrs;
  unsigned owner_id;
} __attribute__((__packed__)) attr_union;

attr_union u;

