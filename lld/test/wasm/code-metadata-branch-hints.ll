# RUN: rm -rf %t; split-file %s %t

# with branch hints
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f1bh.S -o %t/f1.o
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f2bh.S -o %t/f2.o
# RUN: wasm-ld --export-all -o %t.wasm %t/f2.o %t/f1.o
# RUN: obj2yaml %t.wasm | FileCheck --check-prefixes=CHECK %s

# CHECK:          - Type:            CUSTOM
# CHECK:            Name:            metadata.code.branch_hint
# CHECK-NEXT:       Entries:
# CHECK-NEXT:         - FuncIdx:         1
# CHECK-NEXT:           Hints:
# CHECK-NEXT:             - Offset:          7
# CHECK-NEXT:               Size:            1
# CHECK-NEXT:               Data:            UNLIKELY
# CHECK-NEXT:             - Offset:          14
# CHECK-NEXT:               Size:            1
# CHECK-NEXT:               Data:            LIKELY
# CHECK-NEXT:         - FuncIdx:         2
# CHECK-NEXT:           Hints:
# CHECK-NEXT:             - Offset:          5
# CHECK-NEXT:               Size:            1
# CHECK-NEXT:               Data:            LIKELY
# CHECK-NEXT:         - FuncIdx:         3
# CHECK-NEXT:           Hints:
# CHECK-NEXT:             - Offset:          5
# CHECK-NEXT:               Size:            1
# CHECK-NEXT:               Data:            UNLIKELY
# CHECK-NEXT:         - FuncIdx:         4
# CHECK-NEXT:           Hints:
# CHECK-NEXT:             - Offset:          5
# CHECK-NEXT:               Size:            1
# CHECK-NEXT:               Data:            LIKELY

# CHECK:         - Type:            CUSTOM
# CHECK-NEXT:      Name:            name
# CHECK-NEXT:      FunctionNames:
# CHECK-NEXT:        - Index:           0
# CHECK-NEXT:          Name:            __wasm_call_ctors
# CHECK-NEXT:        - Index:           1
# CHECK-NEXT:          Name:            test0
# CHECK-NEXT:        - Index:           2
# CHECK-NEXT:          Name:            test1
# CHECK-NEXT:        - Index:           3
# CHECK-NEXT:          Name:            _start
# CHECK-NEXT:        - Index:           4
# CHECK-NEXT:          Name:            test_func1

# CHECK:        - Type:            CUSTOM
# CHECK:          Name:            target_features
# CHECK-NEXT:     Features:
# CHECK-NEXT:       - Prefix:          USED
# CHECK-NEXT:         Name:            branch-hinting

# without branch hints
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f1.S -o %t/f1.o
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f2.S -o %t/f2.o
# RUN: wasm-ld --export-all -o %t.wasm %t/f2.o %t/f1.o
# RUN: obj2yaml %t.wasm | FileCheck --check-prefixes=NCHECK %s

# NCHECK-NOT:         Name:            metadata.code.branch_hint
# NCHECK-NOT:         Name:            branch-hinting

# with branch hints, but only the _start function is not removed by lld (no --export-all)
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f1bh.S -o %t/f1.o
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mcpu=mvp -filetype=obj %t/f2bh.S -o %t/f2.o
# RUN: wasm-ld -o %t.wasm %t/f2.o %t/f1.o
# RUN: obj2yaml %t.wasm | FileCheck --check-prefixes=RCHECK %s

# RCHECK:          - Type:            CUSTOM
# RCHECK:            Name:            metadata.code.branch_hint
# RCHECK-NEXT:       Entries:
# RCHECK-NEXT:         - FuncIdx:         0
# RCHECK-NEXT:           Hints:
# RCHECK-NEXT:             - Offset:          5
# RCHECK-NEXT:               Size:            1
# RCHECK-NEXT:               Data:            UNLIKELY
# RCHECK-NEXT:    - Type:            CODE

# RCHECK:         - Type:            CUSTOM
# RCHECK-NEXT:      Name:            name
# RCHECK-NEXT:      FunctionNames:
# RCHECK-NEXT:        - Index:           0
# RCHECK-NEXT:          Name:            _start

# RCHECK:        - Type:            CUSTOM
# RCHECK:          Name:            target_features
# RCHECK-NEXT:     Features:
# RCHECK-NEXT:       - Prefix:          USED
# RCHECK-NEXT:         Name:            branch-hinting

#--- f1.S
# Assembly generated based on following ir and
# `llc -mcpu=mvp -filetype=asm ./f1.ll -o f1.S -mattr=-branch-hinting`
# `llc -mcpu=mvp -filetype=asm ./f1.ll -o f1bh.S -mattr=+branch-hinting`
# target triple = "wasm32-unknown-unknown"
#
# define i32 @_start(i32 %a) {
# entry:
#   %cmp = icmp eq i32 %a, 0
#   br i1 %cmp, label %if.then, label %if.else, !prof !0
# if.then:
#   ret i32 1
# if.else:
#   ret i32 2
# }
#
# define i32 @test_func1(i32 %a) {
# entry:
#   %cmp = icmp eq i32 %a, 0
#   br i1 %cmp, label %if.then, label %if.else, !prof !1
# if.then:
#   ret i32 1
# if.else:
#   ret i32 2
# }
#
# !0 = !{!"branch_weights", i32 2000, i32 1}
# !1 = !{!"branch_weights", i32 1, i32 2000}

 .file	"f1.ll"
 .functype	_start (i32) -> (i32)
 .functype	test_func1 (i32) -> (i32)
 .section	.text._start,"",@
 .globl	_start                          # -- Begin function _start
 .type	_start,@function
_start:                                 # @_start
 .functype	_start (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
 br_if   	0                               # 0: down to label0
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB0_2:                                # %if.else
 end_block                               # label0:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.text.test_func1,"",@
 .globl	test_func1                      # -- Begin function test_func1
 .type	test_func1,@function
test_func1:                             # @test_func1
 .functype	test_func1 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
 br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB1_2:                                # %if.else
 end_block                               # label1:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
#--- f1bh.S
 .file	"f1.ll"
 .functype	_start (i32) -> (i32)
 .functype	test_func1 (i32) -> (i32)
 .section	.text._start,"",@
 .globl	_start                          # -- Begin function _start
 .type	_start,@function
_start:                                 # @_start
 .functype	_start (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
.Ltmp0:
 br_if   	0                               # 0: down to label0
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB0_2:                                # %if.else
 end_block                               # label0:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.text.test_func1,"",@
 .globl	test_func1                      # -- Begin function test_func1
 .type	test_func1,@function
test_func1:                             # @test_func1
 .functype	test_func1 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
.Ltmp1:
 br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB1_2:                                # %if.else
 end_block                               # label1:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.custom_section.target_features,"",@
 .int8	1
 .int8	43
 .int8	14
 .ascii	"branch-hinting"
 .section	.text.test_func1,"",@
 .section	.custom_section.metadata.code.branch_hint,"",@
 .int8	2
 .uleb128 _start@FUNCINDEX
 .int8	1
 .uleb128 .Ltmp0
 .int8	1
 .int8	0
 .uleb128 test_func1@FUNCINDEX
 .int8	1
 .uleb128 .Ltmp1
 .int8	1
 .int8	1
 .section	.text.test_func1,"",@

#--- f2.S
# Assembly generated based on following ir and
# `llc -mcpu=mvp -filetype=asm ./f2.ll -o f2.S -mattr=-branch-hinting`
# `llc -mcpu=mvp -filetype=asm ./f2.ll -o f2bh.S -mattr=+branch-hinting`
# target triple = "wasm32-unknown-unknown"
#
# define i32 @test0(i32 %a) {
# entry:
#   %cmp0 = icmp eq i32 %a, 0
#   br i1 %cmp0, label %if.then, label %ret1, !prof !0
# if.then:
#   %cmp1 = icmp eq i32 %a, 1
#   br i1 %cmp1, label %ret1, label %ret2, !prof !1
# ret1:
#   ret i32 2
# ret2:
#   ret i32 1
# }
#
# define i32 @test1(i32 %a) {
# entry:
#   %cmp = icmp eq i32 %a, 0
#   br i1 %cmp, label %if.then, label %if.else, !prof !1
# if.then:
#   ret i32 1
# if.else:
#   ret i32 2
# }
#
# the resulting branch hint is actually reversed, since llvm-br is turned into br_unless, inverting branch probs
# !0 = !{!"branch_weights", i32 2000, i32 1}
# !1 = !{!"branch_weights", i32 1, i32 2000}

 .file	"f2.ll"
 .functype	test0 (i32) -> (i32)
 .functype	test1 (i32) -> (i32)
 .section	.text.test0,"",@
 .globl	test0                           # -- Begin function test0
 .type	test0,@function
test0:                                  # @test0
 .functype	test0 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 block
 local.get	0
 br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
 local.get	0
 i32.const	1
 i32.ne
 br_if   	1                               # 1: down to label0
.LBB0_2:                                # %ret1
 end_block                               # label1:
 i32.const	2
 return
.LBB0_3:                                # %ret2
 end_block                               # label0:
 i32.const	1
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.text.test1,"",@
 .globl	test1                           # -- Begin function test1
 .type	test1,@function
test1:                                  # @test1
 .functype	test1 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
 br_if   	0                               # 0: down to label2
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB1_2:                                # %if.else
 end_block                               # label2:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
#--- f2bh.S
 .file	"f2.ll"
 .functype	test0 (i32) -> (i32)
 .functype	test1 (i32) -> (i32)
 .section	.text.test0,"",@
 .globl	test0                           # -- Begin function test0
 .type	test0,@function
test0:                                  # @test0
 .functype	test0 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 block
 local.get	0
.Ltmp0:
 br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
 local.get	0
 i32.const	1
 i32.ne
.Ltmp1:
 br_if   	1                               # 1: down to label0
.LBB0_2:                                # %ret1
 end_block                               # label1:
 i32.const	2
 return
.LBB0_3:                                # %ret2
 end_block                               # label0:
 i32.const	1
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.text.test1,"",@
 .globl	test1                           # -- Begin function test1
 .type	test1,@function
test1:                                  # @test1
 .functype	test1 (i32) -> (i32)
# %bb.0:                                # %entry
 block
 local.get	0
.Ltmp2:
 br_if   	0                               # 0: down to label2
# %bb.1:                                # %if.then
 i32.const	1
 return
.LBB1_2:                                # %if.else
 end_block                               # label2:
 i32.const	2
                                        # fallthrough-return
 end_function
                                        # -- End function
 .section	.custom_section.target_features,"",@
 .int8	1
 .int8	43
 .int8	14
 .ascii	"branch-hinting"
 .section	.text.test1,"",@
 .section	.custom_section.metadata.code.branch_hint,"",@
 .int8	2
 .uleb128 test0@FUNCINDEX
 .int8	2
 .uleb128 .Ltmp0
 .int8	1
 .int8	0
 .uleb128 .Ltmp1
 .int8	1
 .int8	1
 .uleb128 test1@FUNCINDEX
 .int8	1
 .uleb128 .Ltmp2
 .int8	1
 .int8	1
 .section	.text.test1,"",@
