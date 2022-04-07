// RUN: %clang_cc1 -no-opaque-pointers -fblocks -triple i386-pc-windows-msvc %s -emit-llvm -o - -fblocks | FileCheck %s


int (^x)(void) = ^() { return 21; };


// Check that the block literal is emitted with a null isa pointer
// CHECK: @__block_literal_global = internal global { i8**, i32, i32, i8*, %struct.__block_descriptor* } { i8** null, 

// Check that _NSConcreteGlobalBlock has the correct dllimport specifier.
// CHECK: @_NSConcreteGlobalBlock = external dllimport global i8*
// Check that we create an initialiser pointer in the correct section (early library initialisation).
// CHECK: @.block_isa_init_ptr = internal constant void ()* @.block_isa_init, section ".CRT$XCLa"

// Check that we emit an initialiser for it.
// CHECK: define internal void @.block_isa_init() {
// CHECK: store i8** @_NSConcreteGlobalBlock, i8*** getelementptr inbounds ({ i8**, i32, i32, i8*, %struct.__block_descriptor* }, { i8**, i32, i32, i8*, %struct.__block_descriptor* }* @__block_literal_global, i32 0, i32 0), align 4

