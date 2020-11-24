# REQUIRES: x86
# RUN: llvm-mc -filetype=obj %s -o %t.obj -triple x86_64-windows-msvc
# RUN: lld-link -entry:main -nodefaultlib %t.obj -out:%t.exe -pdb:%t.pdb -debug
# RUN: llvm-symbolizer --obj=%t.exe --use-native-pdb-reader --relative-address \
# RUN:   0x1014 0x1018 0x101c 0x1023 0x1024 | FileCheck %s

# Compiled from this cpp code, with modifications to add extra inline line and 
# file changes:
# clang -cc1 -triple x86_64-windows-msvc -gcodeview -S test.cpp
#
# __attribute__((always_inline)) int inlinee_2(int x) {
#   return x + 1;
# }
# __attribute__((always_inline)) int inlinee_1(int x) {
#   return inlinee_2(x) + 1;
# }
# int main() {
#   return inlinee_1(33);
# }


# CHECK: inlinee_1
# CHECK-NEXT: C:\src\test.cpp:9:0
# CHECK-NEXT: main
# CHECK-NEXT: C:\src\test.cpp:13:10

# CHECK: inlinee_1
# CHECK-NEXT: C:\src\test.cpp:10:0
# CHECK-NEXT: main
# CHECK-NEXT: C:\src\test.cpp:13:10

# CHECK: inlinee_2
# CHECK-NEXT: C:\src\test.cpp:5:0
# CHECK-NEXT: inlinee_1
# CHECK-NEXT: C:\src\test.cpp:9:0
# CHECK-NEXT: main
# CHECK-NEXT: C:\src\test.cpp:13:10

# CHECK: inlinee_2
# CHECK-NEXT: C:\src\file.cpp:5:0
# CHECK-NEXT: inlinee_1
# CHECK-NEXT: C:\src\test.cpp:9:0
# CHECK-NEXT: main
# CHECK-NEXT: C:\src\test.cpp:13:10

# CHECK: inlinee_1
# CHECK-NEXT: C:\src\test.cpp:9:0
# CHECK-NEXT: main
# CHECK-NEXT: C:\src\test.cpp:13:10

  .text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"test.cpp"
	.def	 main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "C:\\src\\test.cpp" "4BECA437CFE062C7D0B74B1851B65988" 1
  .cv_file  2 "C:\\src\\file.cpp" "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF" 1
	.cv_loc	0 1 12 0                        # test.cpp:12:0
# %bb.0:                                # %entry
	subq	$16, %rsp
	movl	$0, 4(%rsp)
	movl	$33, 8(%rsp)
.Ltmp0:
	.cv_inline_site_id 1 within 0 inlined_at 1 13 10
	.cv_loc	1 1 9 20                        # test.cpp:9:20
	movl	8(%rsp), %eax
  .cv_loc 1 1 10 0                        # test.cpp:10:0
	movl	%eax, 12(%rsp)
.Ltmp1:
	.cv_inline_site_id 2 within 1 inlined_at 1 9 10
	.cv_loc	2 1 5 10                        # test.cpp:5:10
	movl	12(%rsp), %eax
	.cv_loc	2 1 5 12                        # test.cpp:5:12
	addl	$1, %eax
  .cv_loc 2 2 5 13                        # file.cpp:5:13
  nop
.Ltmp2:
	.cv_loc	1 1 9 23                        # test.cpp:9:23
	addl	$1, %eax
.Ltmp3:
	.cv_loc	0 1 13 3                        # test.cpp:13:3
	addq	$16, %rsp
	retq
.Ltmp4:
.Lfunc_end0:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp6-.Ltmp5                   # Subsection size
.Ltmp5:
	.short	.Ltmp8-.Ltmp7                   # Record length
.Ltmp7:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
	.short	12                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	12000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 12.0.0 (https://github.com/llvm/llvm-project.git 6a4850e9c1cc74cc67f99f1f81a8fe060a7088d2)" # Null-terminated compiler version string
	.p2align	2
.Ltmp8:
.Ltmp6:
	.p2align	2
	.long	246                             # Inlinee lines subsection
	.long	.Ltmp10-.Ltmp9                  # Subsection size
.Ltmp9:
	.long	0                               # Inlinee lines signature

                                        # Inlined function inlinee_1 starts at test.cpp:8
	.long	4098                            # Type index of inlined function
	.cv_filechecksumoffset	1               # Offset into filechecksum table
	.long	8                               # Starting line number

                                        # Inlined function inlinee_2 starts at test.cpp:4
	.long	4099                            # Type index of inlined function
	.cv_filechecksumoffset	1               # Offset into filechecksum table
	.long	4                               # Starting line number
.Ltmp10:
	.p2align	2
	.long	241                             # Symbol subsection for main
	.long	.Ltmp12-.Ltmp11                 # Subsection size
.Ltmp11:
	.short	.Ltmp14-.Ltmp13                 # Record length
.Ltmp13:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4102                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
.Ltmp14:
	.short	.Ltmp16-.Ltmp15                 # Record length
.Ltmp15:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	16                              # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	81920                           # Flags (defines frame register)
	.p2align	2
.Ltmp16:
	.short	.Ltmp18-.Ltmp17                 # Record length
.Ltmp17:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4098                            # Inlinee type index
	.cv_inline_linetable	1 1 8 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp18:
	.short	.Ltmp20-.Ltmp19                 # Record length
.Ltmp19:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
.Ltmp20:
	.cv_def_range	 .Ltmp0 .Ltmp3, frame_ptr_rel, 8
	.short	.Ltmp22-.Ltmp21                 # Record length
.Ltmp21:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4099                            # Inlinee type index
	.cv_inline_linetable	2 1 4 .Lfunc_begin0 .Lfunc_end0
	.p2align	2
.Ltmp22:
	.short	.Ltmp24-.Ltmp23                 # Record length
.Ltmp23:
	.short	4414                            # Record kind: S_LOCAL
	.long	116                             # TypeIndex
	.short	1                               # Flags
	.asciz	"x"
	.p2align	2
.Ltmp24:
	.cv_def_range	 .Ltmp1 .Ltmp2, frame_ptr_rel, 12
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp12:
	.p2align	2
	.cv_linetable	0, main, .Lfunc_end0
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp26-.Ltmp25                 # Subsection size
.Ltmp25:
	.short	.Ltmp28-.Ltmp27                 # Record length
.Ltmp27:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4105                            # LF_BUILDINFO index
	.p2align	2
.Ltmp28:
.Ltmp26:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# ArgList (0x1000)
	.short	0xa                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x1                             # NumArgs
	.long	0x74                            # Argument: int
	# Procedure (0x1001)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x1                             # NumParameters
	.long	0x1000                          # ArgListType: (int)
	# FuncId (0x1002)
	.short	0x16                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: int (int)
	.asciz	"inlinee_1"                     # Name
	.byte	242
	.byte	241
	# FuncId (0x1003)
	.short	0x16                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: int (int)
	.asciz	"inlinee_2"                     # Name
	.byte	242
	.byte	241
	# ArgList (0x1004)
	.short	0x6                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x0                             # NumArgs
	# Procedure (0x1005)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1004                          # ArgListType: ()
	# FuncId (0x1006)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1005                          # FunctionType: int ()
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1007)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"C:\\src"                       # StringData
	.byte	241
	# StringId (0x1008)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"<stdin>"                       # StringData
	# BuildInfo (0x1009)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x1007                          # Argument: C:\src
	.long	0x0                             # Argument
	.long	0x1008                          # Argument: <stdin>
	.long	0x0                             # Argument
	.long	0x0                             # Argument
	.byte	242
	.byte	241
