// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s
// CHECK: {{ERR_DNS_SERVER_REQUIRES_TCP$}}

// http://llvm.org/PR12674
#define NET_ERROR(label, value) ERR_ ## label = value,

NET_ERROR(DNS_SERVER_REQUIRES_TCP, -801)

#undef NET_ERROR

