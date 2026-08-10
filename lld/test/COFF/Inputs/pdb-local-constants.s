	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"local_constant.cpp"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:i <- 123
	.cv_file	1 "/home/tobias/code/llvm-project/build/local_constant.cpp" "C33315002D9B48E67EB3E617E430BC02" 1
	.cv_loc	0 1 7 0                         # local_constant.cpp:7:0
	movl	$444, %eax                      # imm = 0x1BC
	retq
.Ltmp0:
.Lfunc_end0:
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp2-.Ltmp1                   # Subsection size
.Ltmp1:
	.short	.Ltmp4-.Ltmp3                   # Record length
.Ltmp3:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.byte	0                               # Object name
	.p2align	2, 0x0
.Ltmp4:
	.short	.Ltmp6-.Ltmp5                   # Record length
.Ltmp5:
	.short	4412                            # Record kind: S_COMPILE3
	.long	1                               # Flags and language
	.short	208                             # CPUType
	.short	16                              # Frontend version
	.short	0
	.short	0
	.short	0
	.short	16000                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 16.0.0 (git@github.com:llvm/llvm-project.git eef89bd2b3f4a13efcad176bb4c4dda1b1e202ce)" # Null-terminated compiler version string
	.p2align	2, 0x0
.Ltmp6:
.Ltmp2:
	.p2align	2, 0x0
	.long	241                             # Symbol subsection for main
	.long	.Ltmp8-.Ltmp7                   # Subsection size
.Ltmp7:
	.short	.Ltmp10-.Ltmp9                  # Record length
.Ltmp9:
	.short	4423                            # Record kind: S_GPROC32_ID
	.long	0                               # PtrParent
	.long	0                               # PtrEnd
	.long	0                               # PtrNext
	.long	.Lfunc_end0-main                # Code size
	.long	0                               # Offset after prologue
	.long	0                               # Offset before epilogue
	.long	4098                            # Function type index
	.secrel32	main                    # Function section relative address
	.secidx	main                            # Function section index
	.byte	0                               # Flags
	.asciz	"main"                          # Function name
	.p2align	2, 0x0
.Ltmp10:
	.short	.Ltmp12-.Ltmp11                 # Record length
.Ltmp11:
	.short	4114                            # Record kind: S_FRAMEPROC
	.long	0                               # FrameSize
	.long	0                               # Padding
	.long	0                               # Offset of padding
	.long	0                               # Bytes of callee saved registers
	.long	0                               # Exception handler offset
	.short	0                               # Exception handler section
	.long	1056768                         # Flags (defines frame register)
	.p2align	2, 0x0
.Ltmp12:
	.short	.Ltmp14-.Ltmp13                 # Record length
.Ltmp13:
	.short	4359                            # Record kind: S_CONSTANT
	.long	116                             # Type
	.byte	0x7b, 0x00                      # Value
	.asciz	"i"                             # Name
	.p2align	2, 0x0
.Ltmp14:
	.short	2                               # Record length
	.short	4431                            # Record kind: S_PROC_ID_END
.Ltmp8:
	.p2align	2, 0x0
	.cv_linetable	0, main, .Lfunc_end0
	.long	241                             # Symbol subsection for globals
	.long	.Ltmp16-.Ltmp15                 # Subsection size
.Ltmp15:
	.short	.Ltmp18-.Ltmp17                 # Record length
.Ltmp17:
	.short	4359                            # Record kind: S_CONSTANT
	.long	4099                            # Type
	.byte	0x41, 0x01                      # Value
	.asciz	"g_const"                       # Name
	.p2align	2, 0x0
.Ltmp18:
.Ltmp16:
	.p2align	2, 0x0
	.cv_filechecksums                       # File index to string table offset subsection
	.cv_stringtable                         # String table
	.long	241
	.long	.Ltmp20-.Ltmp19                 # Subsection size
.Ltmp19:
	.short	.Ltmp22-.Ltmp21                 # Record length
.Ltmp21:
	.short	4428                            # Record kind: S_BUILDINFO
	.long	4103                            # LF_BUILDINFO index
	.p2align	2, 0x0
.Ltmp22:
.Ltmp20:
	.p2align	2, 0x0
	.section	.debug$T,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	# ArgList (0x1000)
	.short	0x6                             # Record length
	.short	0x1201                          # Record kind: LF_ARGLIST
	.long	0x0                             # NumArgs
	# Procedure (0x1001)
	.short	0xe                             # Record length
	.short	0x1008                          # Record kind: LF_PROCEDURE
	.long	0x74                            # ReturnType: int
	.byte	0x0                             # CallingConvention: NearC
	.byte	0x0                             # FunctionOptions
	.short	0x0                             # NumParameters
	.long	0x1000                          # ArgListType: ()
	# FuncId (0x1002)
	.short	0x12                            # Record length
	.short	0x1601                          # Record kind: LF_FUNC_ID
	.long	0x0                             # ParentScope
	.long	0x1001                          # FunctionType: int ()
	.asciz	"main"                          # Name
	.byte	243
	.byte	242
	.byte	241
	# Modifier (0x1003)
	.short	0xa                             # Record length
	.short	0x1001                          # Record kind: LF_MODIFIER
	.long	0x74                            # ModifiedType: int
	.short	0x1                             # Modifiers ( Const (0x1) )
	.byte	242
	.byte	241
	# StringId (0x1004)
	.short	0x2e                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"/home/tobias/code/llvm-project/build" # StringData
	.byte	243
	.byte	242
	.byte	241
	# StringId (0x1005)
	.short	0x1a                            # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.asciz	"local_constant.cpp"            # StringData
	.byte	241
	# StringId (0x1006)
	.short	0xa                             # Record length
	.short	0x1605                          # Record kind: LF_STRING_ID
	.long	0x0                             # Id
	.byte	0                               # StringData
	.byte	243
	.byte	242
	.byte	241
	# BuildInfo (0x1007)
	.short	0x1a                            # Record length
	.short	0x1603                          # Record kind: LF_BUILDINFO
	.short	0x5                             # NumArgs
	.long	0x1004                          # Argument: /home/tobias/code/llvm-project/build
	.long	0x0                             # Argument
	.long	0x1005                          # Argument: local_constant.cpp
	.long	0x1006                          # Argument
	.long	0x0                             # Argument
	.byte	242
	.byte	241
	.addrsig
