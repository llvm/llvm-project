// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm  \
// RUN:   -mcpu=pwr10 -mpaired-vector-memops %s -o - | FileCheck %s \
// RUN:   --check-prefix=HASPAIRED
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mpaired-vector-memops %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPAIRED
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mpaired-vector-memops %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPAIRED

// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm  \
// RUN:   -mcpu=pwr10 -mprefixed %s -o - | FileCheck %s \
// RUN:   --check-prefix=HASPREFIXED
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mprefixed %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPREFIXED
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mprefixed %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPREFIXED

// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm  \
// RUN:   -mcpu=pwr10 -mpcrel %s -o - | FileCheck %s \
// RUN:   --check-prefix=HASPCREL
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mpcrel %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPCREL
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mpcrel %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPCREL

// RUN: %clang -target powerpc64le-unknown-linux-gnu -S -emit-llvm  \
// RUN:   -mcpu=pwr10 -mpcrel -mprefixed %s -o - | FileCheck %s \
// RUN:   --check-prefix=HASPCREL-PREFIX
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mpcrel -mprefixed %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPCREL-PREFIX
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mpcrel -mprefixed %s 2>&1 | FileCheck %s \
// RUN:   --check-prefix=NOPCREL-PREFIX

int test_p10_features() {
  return 0;
}

// HASPAIRED: test_p10_features() #0 {
// HASPAIRED: attributes #0 = {
// HASPAIRED-SAME: +paired-vector-memops
// NOPAIRED: option '-mpaired-vector-memops' cannot be specified without '-mcpu=pwr10'

// HASPREFIXED: test_p10_features() #0 {
// HASPREFIXED: attributes #0 = {
// HASPREFIXED-SAME: +prefix-instrs
// NOPREFIXED: option '-mprefixed' cannot be specified without '-mcpu=pwr10'

// HASPCREL: test_p10_features() #0 {
// HASPCREL: attributes #0 = {
// HASPCREL-SAME: +pcrelative-memops
// NOPCREL: option '-mpcrel' cannot be specified without '-mcpu=pwr10 -mprefixed'

// HASPCREL-PREFIX: test_p10_features() #0 {
// HASPCREL-PREFIX: attributes #0 = {
// HASPCREL-PREFIX-SAME: +pcrelative-memops
// HASPCREL-PREFIX-SAME: +prefix-instrs
// NOPCREL-PREFIX: option '-mpcrel' cannot be specified without '-mcpu=pwr10 -mprefixed'

