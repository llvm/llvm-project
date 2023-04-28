// RUN: %clang_cc1 -E -P -H %s 2>&1 | grep "multiple-inclusion-opt.h" | count 1

#include "multiple-inclusion-opt.h"
#include "multiple-inclusion-opt.h"
#include "multiple-inclusion-opt.h"
#include "multiple-inclusion-opt.h"
#include "multiple-inclusion-opt.h"
