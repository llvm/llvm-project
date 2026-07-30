// RUN: %clang_cc1 -triple powerpc-ibm-aix   -S -fcommon %s -verify -o -
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -S -fcommon %s -verify -o -
int xxxxxx;
extern int yyyyyy __attribute__((__alias__("xxxxxx") )); //expected-error {{alias to a variable in a common section is not allowed}}

void *gggggg() { return &yyyyyy; }
