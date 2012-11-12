// RUN: %clang_cc1 -fblocks -g -emit-llvm -o - %s | FileCheck %s
// APPLE LOCAL file 5939894 */
// Verify that the desired debugging type is generated for a structure
//  member that is a pointer to a block. 

// CHECK: __block_literal_generic{{.*}}DW_TAG_structure_type
// CHECK: __block_descriptor{{.*}}DW_TAG_structure_type
struct inStruct {
  void (^genericBlockPtr)();
} is;

