// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fdump-record-layouts %s -o - | FileCheck %s

struct SimpleStruct {
  int a;
  float b;
} simple;
// CHECK: Layout: <CIRecordLayout
// CHECK: CIR Type:!cir.record<struct "SimpleStruct" {!cir.int<s, 32>, !cir.float} #cir.record.decl.ast>
// CHECK: NonVirtualBaseCIRType:!cir.record<struct "SimpleStruct" {!cir.int<s, 32>, !cir.float} #cir.record.decl.ast>
// CHECK: IsZeroInitializable:1
// CHECK:   BitFields:[
// CHECK: ]>

struct Empty {
} empty;

// CHECK: Layout: <CIRecordLayout
// CHECK:  CIR Type:!cir.record<struct "Empty" padded {!cir.int<u, 8>} #cir.record.decl.ast>
// CHECK:  NonVirtualBaseCIRType:!cir.record<struct "Empty" padded {!cir.int<u, 8>} #cir.record.decl.ast>
// CHECK:  IsZeroInitializable:1
// CHECK:  BitFields:[
// CHECK:  ]>

struct BitfieldsInOrder {
  char a;
  unsigned bit: 8;
  unsigned should : 20;
  unsigned have: 3;
  unsigned order: 1;
} bitfield_order;

// CHECK: Layout: <CIRecordLayout
// CHECK:  CIR Type:!cir.record<struct "BitfieldsInOrder" {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<u, 32>} #cir.record.decl.ast>
// CHECK:  NonVirtualBaseCIRType:!cir.record<struct "BitfieldsInOrder" {!cir.int<s, 8>, !cir.int<u, 8>, !cir.int<u, 32>} #cir.record.decl.ast>
// CHECK:  IsZeroInitializable:1
// CHECK:  BitFields:[
// CHECK-NEXT:   <CIRBitFieldInfo name:bit offset:0 size:8 isSigned:0 storageSize:8 storageOffset:1 volatileOffset:0 volatileStorageSize:0 volatileStorageOffset:0>
// CHECK-NEXT:   <CIRBitFieldInfo name:should offset:0 size:20 isSigned:0 storageSize:32 storageOffset:4 volatileOffset:0 volatileStorageSize:0 volatileStorageOffset:0>
// CHECK-NEXT:   <CIRBitFieldInfo name:have offset:20 size:3 isSigned:0 storageSize:32 storageOffset:4 volatileOffset:0 volatileStorageSize:0 volatileStorageOffset:0>
// CHECK-NEXT:   <CIRBitFieldInfo name:order offset:23 size:1 isSigned:0 storageSize:32 storageOffset:4 volatileOffset:0 volatileStorageSize:0 volatileStorageOffset:0>
// CHECK:]>

struct Inner {
  int x;
} in;

//CHECK: Layout: <CIRecordLayout
//CHECK:  CIR Type:!cir.record<struct "Inner" {!cir.int<s, 32>} #cir.record.decl.ast>
//CHECK:  NonVirtualBaseCIRType:!cir.record<struct "Inner" {!cir.int<s, 32>} #cir.record.decl.ast>
//CHECK:  IsZeroInitializable:1
//CHECK:  BitFields:[
//CHECK:  ]>

struct Outer {
  Inner i;
  int y = 6;
} ou;

//CHECK: Layout: <CIRecordLayout
//CHECK:  CIR Type:!cir.record<struct "Outer" {!cir.record<struct "Inner" {!cir.int<s, 32>} #cir.record.decl.ast>, !cir.int<s, 32>} #cir.record.decl.ast>
//CHECK:  NonVirtualBaseCIRType:!cir.record<struct "Outer" {!cir.record<struct "Inner" {!cir.int<s, 32>} #cir.record.decl.ast>, !cir.int<s, 32>} #cir.record.decl.ast>
//CHECK:  IsZeroInitializable:1
//CHECK:  BitFields:[
//CHECK:  ]>
