# REQUIRES: x86
# RUN: llvm-mc -filetype=obj %s -o %t.obj -triple x86_64-windows-msvc
# RUN: lld-link -entry:main -nodefaultlib %t.obj -out:%t.exe -pdb:%t.pdb -debug
# RUN: llvm-symbolizer --obj=%t.exe --relative-address \
# RUN:   0x1000 0x1003 0x1010 0x1013 | FileCheck %s

# Compiled from this cpp code:
# int f1(int x) {
#   int y = x + 1;
#   return y;
# }
# int f2(int n) {
#   return f1(n);
# }
# int main() {
#   return f2(100);
# }

.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"t.cpp"
	.def	 "?f1@@YAHH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	"?f1@@YAHH@Z"                   # -- Begin function ?f1@@YAHH@Z
	.p2align	4, 0x90
"?f1@@YAHH@Z":                          # @"?f1@@YAHH@Z"
.Lfunc_begin0:
	.cv_func_id 0
# %bb.0:                                # %entry
	.cv_file	1 "C:\\src\\tests\\t.cpp" "E6E6D87A9021656AD44E74484F5BA421" 1

# CHECK:      f1(int)
# CHECK-NEXT: t.cpp:2:13
	.cv_loc	0 1 2 13                        # t.cpp:2:13
                                        # kill: def $ecx killed $ecx def $rcx
	leal	1(%rcx), %eax

# CHECK:      f1(int)
# CHECK-NEXT: t.cpp:3:3
	.cv_loc	0 1 3 3                         # t.cpp:3:3
	retq
.Ltmp0:
.Lfunc_end0:
                                        # -- End function
	.def	 "?f2@@YAHH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	"?f2@@YAHH@Z"                   # -- Begin function ?f2@@YAHH@Z
	.p2align	4, 0x90
"?f2@@YAHH@Z":                          # @"?f2@@YAHH@Z"
.Lfunc_begin1:
	.cv_func_id 1
# %bb.0:                                # %entry
# CHECK:      f1
# CHECK-NEXT: t.cpp:2:0
# CHECK-NEXT: f2(int)
# CHECK-NEXT: t.cpp:6:3
	.cv_inline_site_id 2 within 1 inlined_at 1 6 10
	.cv_loc	2 1 2 13                        # t.cpp:2:13
                                        # kill: def $ecx killed $ecx def $rcx
	leal	1(%rcx), %eax
.Ltmp1:
	.cv_loc	1 1 6 3                         # t.cpp:6:3
	retq
# CHECK:      f2(int)
# CHECK-NEXT: t.cpp:6:3
.Ltmp2:
.Lfunc_end1:
                                        # -- End function
	.def	 main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin2:
	.cv_func_id 3
# %bb.0:                                # %entry
	.cv_loc	3 1 9 3                         # t.cpp:9:3
	movl	$101, %eax
	retq
.Ltmp3:
.Lfunc_end2:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp5-.Ltmp4                   # Subsection size
.Ltmp4:
	.short	.Ltmp7-.Ltmp6                   # Record length
.Ltmp6:
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
	.asciz	"clang version 12.0.0 (https://github.com/llvm/llvm-project.git e2e86f4e77ec2fd79743f4d0e94689e9668600ad)" # Null-terminated compiler version string
	.p2align	2
.Ltmp7:
.Ltmp5:
	.p2align	2
	.long	246                             # Inlinee lines subsection
	.long	.Ltmp9-.Ltmp8                   # Subsection size
.Ltmp8:
	.long	0                               # Inlinee lines signature

                                        # Inlined function f1 starts at t.cpp:1
	.long	4098                            # Type index of inlined function
	.cv_filechecksumoffset	1               # Offset into filechecksum table
	.long	1                               # Starting line number
.Ltmp9:
	.p2align	2
	.long	241                             # Symbol subsection for f1
	.long	.Ltmp11-.Ltmp10                 # Subsection size
.Ltmp10:
	.short	.Ltmp13-.Ltmp12                 # Record length
.Ltmp12:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-"?f1@@YAHH@Z"       # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4098                            # Function type index
	.secrel32	"?f1@@YAHH@Z"           # Function section relative address
	.secidx	"?f1@@YAHH@Z"                   # Function section index
	.byte	0                               # Flags
	.asciz	"f1"                            # Function name
	.p2align	2
.Ltmp13:
	.short	.Ltmp15-.Ltmp14                 # Record length
.Ltmp14:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	1048576                         # Flags (defines frame register)
	.p2align	2
.Ltmp15:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp11:
	.p2align	2
	.cv_linetable	0, "?f1@@YAHH@Z", .Lfunc_end0
	.long	241                             # Symbol subsection for f2
	.long	.Ltmp17-.Ltmp16                 # Subsection size
.Ltmp16:
	.short	.Ltmp19-.Ltmp18                 # Record length
.Ltmp18:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end1-"?f2@@YAHH@Z"       # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4099                            # Function type index
	.secrel32	"?f2@@YAHH@Z"           # Function section relative address
	.secidx	"?f2@@YAHH@Z"                   # Function section index
	.byte	0                               # Flags
	.asciz	"f2"                            # Function name
	.p2align	2
.Ltmp19:
	.short	.Ltmp21-.Ltmp20                 # Record length
.Ltmp20:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	1048576                         # Flags (defines frame register)
	.p2align	2
.Ltmp21:
	.short	.Ltmp23-.Ltmp22                 # Record length
.Ltmp22:
	.short	4429                            # Record kind: S_INLINESITE
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	4098                            # Inlinee type index
	.cv_inline_linetable	2 1 1 .Lfunc_begin1 .Lfunc_end1
	.p2align	2
.Ltmp23:
	.short	2                               # Record length
	.short	4430                            # Record kind: S_INLINESITE_END
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp17:
	.p2align	2
	.cv_linetable	1, "?f2@@YAHH@Z", .Lfunc_end1
	.long	241                             # Symbol subsection for main
	.long	.Ltmp25-.Ltmp24                 # Subsection size
.Ltmp24:
	.short	.Ltmp27-.Ltmp26                 # Record length
.Ltmp26:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end2-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4100                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2
.Ltmp27:
	.short	.Ltmp29-.Ltmp28                 # Record length
.Ltmp28:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	1048576                         # Flags (defines frame register)
	.p2align	2
.Ltmp29:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp25:
	.p2align	2
	.cv_linetable	3, main, .Lfunc_end2
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp31-.Ltmp30                 # Subsection size
.Ltmp30:
	.short	.Ltmp33-.Ltmp32                 # Record length
.Ltmp32:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4103                            # LF_BUILDINFO index
	.p2align	2
.Ltmp33:
.Ltmp31:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                               # Debug section magic
	# ArgList (0x1000)
	.short	0x6                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x0                             # NumArgs
	# Procedure (0x1001)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x3                             # ReturnType: void
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1000                          # ArgListType: ()
	# FuncId (0x1002)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"f1"                            # Name
	.byte	241
	# FuncId (0x1003)
	.short	0xe                             # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"f2"                            # Name
	.byte	241
	# FuncId (0x1004)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: void ()
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1005)
	.short	0x16                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"C:\\src\\tests"                # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1006)
	.short	0xe                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"<stdin>"                       # StringData
	# BuildInfo (0x1007)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x1005                          # Argument: C:\src\tests
	.long	0x0                             # Argument
	.long	0x1006                          # Argument: <stdin>
	.long	0x0                             # Argument
	.long	0x0                             # Argument
	.byte	242
	.byte	241
