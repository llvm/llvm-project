// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s  -o /dev/null

static unsigned char out[]={0,1};
static const unsigned char str1[]="1";

