// RUN: %clang_analyze_cc1 -isystem %S/Inputs/ -verify %s \
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete

// RUN: %clang_analyze_cc1 -I %S/Inputs/ -verify %s \
// RUN:   -analyzer-checker=core,unix.Malloc,cplusplus.NewDelete

#include "overloaded-delete-in-header.h"

void deleteInHeader(DeleteInHeader *p) { delete p; }
