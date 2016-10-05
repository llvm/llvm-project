# REQUIRES: x86

## The test file contains a STT_TLS symbol but has no TLS section.
# RUN: not ld.lld %S/Inputs/tls-symbol.elf -o %t 2>&1 | FileCheck %s
# CHECK: has a STT_TLS symbol but doesn't have a PT_TLS section
