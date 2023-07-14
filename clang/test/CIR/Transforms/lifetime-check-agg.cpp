// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

typedef enum SType {
  INFO_ENUM_0 = 9,
  INFO_ENUM_1 = 2020,
} SType;

typedef struct InfoRaw {
    SType type;
    const void* __attribute__((__may_alias__)) next;
} InfoRaw;

typedef unsigned long long FlagsPriv;
typedef struct InfoPriv {
    SType type;
    void* __attribute__((__may_alias__)) next;
    FlagsPriv flags;
} InfoPriv;

static const FlagsPriv PrivBit = 0x00000001;

void escape_info(InfoRaw *info);
void exploded_fields(bool cond) {
  {
    InfoRaw info = {INFO_ENUM_0};
    if (cond) {
      InfoPriv privTmp = {INFO_ENUM_1};
      privTmp.flags = PrivBit;
      info.next = &privTmp;
    }
    escape_info(&info);
  }
}