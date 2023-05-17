// RUN: %check_clang_tidy %s \
// RUN: cppcoreguidelines-pro-type-member-init %t \
// RUN: -config="{CheckOptions: \
// RUN: [{key: cppcoreguidelines-pro-type-member-init.IgnoreArrays, value: true} ]}"

typedef int TypedefArray[4];
using UsingArray = int[4];

struct HasArrayMember {
  HasArrayMember() {}
  // CHECK-MESSAGES: warning: constructor does not initialize these fields: Number
  UsingArray U;
  TypedefArray T;
  int RawArray[4];
  int Number;
};
