// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs -isystem %S/Inputs/nullability-consistency-system %s -verify
// RUN: %clang_cc1 -fsyntax-only -fblocks -I %S/Inputs -isystem %S/Inputs/nullability-consistency-system %s -Wsystem-headers -DWARN_IN_SYSTEM_HEADERS -verify

#include "nullability-consistency-1.h"
#include "nullability-consistency-3.h"
#include "nullability-consistency-4.h"
#include "nullability-consistency-5.h"
#include "nullability-consistency-5.h"
#include "nullability-consistency-6.h"
#include "nullability-consistency-7.h"
#include "nullability-consistency-8.h"
#include "nullability-consistency-system.h"

void h1(int *ptr) { } // don't warn

void h2(int * _Nonnull) { }
