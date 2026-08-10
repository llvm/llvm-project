// RUN: %clang_cc1 -fblocks -triple i386-pc-windows-msvc %s -emit-llvm -o - -fblocks | FileCheck %s


int (^x)(void) = ^() { return 21; };


// Check that the block literal is emitted with a null isa pointer
// CHECK: @__block_literal_global = internal global { ptr, i32, i32, ptr, ptr } { ptr null, 

// Check that _NSConcreteGlobalBlock has the correct dllimport specifier.
// CHECK: @_NSConcreteGlobalBlock = external dllimport global ptr
// Check that we create an initialiser pointer in the correct section (early library initialisation).
// CHECK: @.block_isa_init_ptr = internal constant ptr @.block_isa_init, section ".CRT$XCLa"

// Check that we emit an initialiser for it.
// CHECK: define internal void @.block_isa_init() {
// CHECK: store ptr @_NSConcreteGlobalBlock, ptr @__block_literal_global, align 4

