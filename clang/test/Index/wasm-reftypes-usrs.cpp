// REQUIRES: webassembly-registered-target
// RUN: c-index-test -test-load-source-usrs all -target wasm32 -mreference-types %s | FileCheck %s

void take(__externref_t);

// CHECK: wasm-reftypes-usrs.cpp c:@F@take#@BT@__externref_t# Extent=[4:1 - 4:25]
