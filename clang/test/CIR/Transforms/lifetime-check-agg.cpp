// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all;remarks=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

typedef enum SType {
  INFO_ENUM_0 = 9,
  INFO_ENUM_1 = 2020,
} SType;

typedef struct InfoRaw {
    SType type;
    const void* __attribute__((__may_alias__)) next;
    unsigned int fa;
    unsigned f;
    unsigned s;
    unsigned w;
    unsigned h;
    unsigned g;
    unsigned a;
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
    InfoRaw info = {INFO_ENUM_0}; // expected-note {{invalidated here}}
    if (cond) {
      InfoPriv privTmp = {INFO_ENUM_1};
      privTmp.flags = PrivBit;
      info.next = &privTmp;
    } // expected-note {{pointee 'privTmp' invalidated at end of scope}}

    // If the 'if' above is taken, info.next is invalidated at the end of the scope, otherwise
    // it's also invalid because it was initialized with 'nullptr'. This could be a noisy
    // check if calls like `escape_info` are used to further initialize `info`.

    escape_info(&info); // expected-remark {{pset => { invalid, nullptr }}}
                        // expected-warning@-1 {{passing aggregate containing invalid pointer member 'info.next'}}
  }
}

void exploded_fields1(bool cond, unsigned t) {
  {
    InfoRaw info = {INFO_ENUM_0, &t};
    if (cond) {
      InfoPriv privTmp = {INFO_ENUM_1};
      privTmp.flags = PrivBit;
      info.next = &privTmp;
    }

    // A warning is not emitted here, lack of context for inferring
    // anything about `cond` would make it too noisy given `info.next`
    // wasn't null initialized.

    escape_info(&info); // expected-remark {{pset => { t }}}
  }
}
