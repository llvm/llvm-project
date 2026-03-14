// RUN: c-index-test core -print-source-symbols -- %s -isystem %S/Inputs/sys | FileCheck %S/Inputs/sys/system-head.h

#include "system-head.h"
