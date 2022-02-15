// RUN: %clang_cc1 -triple armv7-apple-darwin -target-feature +neon %s -emit-llvm -o - | FileCheck %s

typedef struct _zend_ini_entry zend_ini_entry;
struct _zend_ini_entry {
  void *mh_arg1;
};

char a;

const zend_ini_entry ini_entries[] = {
  {  ((char*)&((zend_ini_entry*)0)->mh_arg1 - (char*)(void*)0)},
};

// PR7564
struct GLGENH {
  int : 27;
  int EMHJAA : 1;
};

struct GLGENH ABHFBF = {1};

typedef __attribute__(( ext_vector_type(2) )) unsigned int uint2;
typedef __attribute__(( __vector_size__(8) )) unsigned int __neon_uint32x2_t;

// rdar://8183908
typedef unsigned int uint32_t;
typedef __attribute__((neon_vector_type(2)))  uint32_t uint32x2_t;
void foo(void) {
    const uint32x2_t signBit = { (uint2) 0x80000000 };
}

// CHECK: %struct.fp_struct_foo = type { void ([1 x i32])* }
struct fp_struct_bar { int a; };

struct fp_struct_foo {
  void (*FP)(struct fp_struct_bar);
} G;
