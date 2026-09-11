// REQUIRES: systemz-registered-target
//RUN: %clang -c  --target=s390x-ibm-zos %s -mzos-sys-include=%S/Inputs/zos/usr/include -Xclang -verify

// expected-no-diagnostics

#include <grp.h>
#include <locale.h>
#include <math.h>
#include <poll.h>
#include <string.h>
#include <time.h>
#include <variant.h>

int __grp;
int __locale;
int __math;
int __poll;
int __string;
int __time;
int __variant;
