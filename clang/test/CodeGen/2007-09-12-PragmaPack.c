// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

#pragma pack(push, 1)
typedef struct
{
        uint32_t        a;
} foo;

typedef struct {
        uint8_t         major;
        uint8_t         minor;
        uint16_t        build;
} VERSION;

typedef struct {
        uint8_t       a[5];
        VERSION       version;
        uint8_t       b;
        foo           d;
        uint32_t      guard;
} bar;
#pragma pack(pop)


unsigned barsize(void) {
  // CHECK: ret i32 18
  return sizeof(bar);
}
