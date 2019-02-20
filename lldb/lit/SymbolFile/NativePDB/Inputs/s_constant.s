	.text
	.def	 @feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.intel_syntax noprefix
	.def	 main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.Lfunc_begin0:
	.cv_func_id 0
	.cv_file	1 "D:\\src\\llvm-mono\\lldb\\lit\\SymbolFile\\NativePDB\\s_constant.cpp" "7F1DA683A9B72A1360C1FDEDD7550E06" 1
	.cv_loc	0 1 79 0                # D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp:79:0
.seh_proc main
# %bb.0:                                # %entry
	sub	rsp, 24
	.seh_stackalloc 24
	.seh_endprologue
	xor	eax, eax
	mov	dword ptr [rsp + 20], 0
	mov	qword ptr [rsp + 8], rdx
	mov	dword ptr [rsp + 4], ecx
.Ltmp0:
	.cv_loc	0 1 80 0                # D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp:80:0
	add	rsp, 24
	ret
.Ltmp1:
.Lfunc_end0:
	.seh_handlerdata
	.text
	.seh_endproc
                                        # -- End function
	.section	.rdata,"dr"
	.p2align	3               # @GlobalLUEA
GlobalLUEA:
	.quad	0                       # 0x0

	.p2align	3               # @GlobalLUEB
GlobalLUEB:
	.quad	1000                    # 0x3e8

	.p2align	3               # @GlobalLUEC
GlobalLUEC:
	.quad	-16                     # 0xfffffffffffffff0

	.p2align	3               # @GlobalLSEA
GlobalLSEA:
	.quad	0                       # 0x0

	.p2align	3               # @GlobalLSEB
GlobalLSEB:
	.quad	9223372036854775000     # 0x7ffffffffffffcd8

	.p2align	3               # @GlobalLSEC
GlobalLSEC:
	.quad	-9223372036854775000    # 0x8000000000000328

	.p2align	2               # @GlobalUEA
GlobalUEA:
	.long	0                       # 0x0

	.p2align	2               # @GlobalUEB
GlobalUEB:
	.long	1000                    # 0x3e8

	.p2align	2               # @GlobalUEC
GlobalUEC:
	.long	4294000000              # 0xfff13d80

	.p2align	2               # @GlobalSEA
GlobalSEA:
	.long	0                       # 0x0

	.p2align	2               # @GlobalSEB
GlobalSEB:
	.long	2147000000              # 0x7ff89ec0

	.p2align	2               # @GlobalSEC
GlobalSEC:
	.long	2147967296              # 0x80076140

GlobalSUEA:                             # @GlobalSUEA
	.byte	0                       # 0x0

GlobalSUEB:                             # @GlobalSUEB
	.byte	100                     # 0x64

GlobalSUEC:                             # @GlobalSUEC
	.byte	200                     # 0xc8

GlobalSSEA:                             # @GlobalSSEA
	.byte	0                       # 0x0

GlobalSSEB:                             # @GlobalSSEB
	.byte	100                     # 0x64

GlobalSSEC:                             # @GlobalSSEC
	.byte	156                     # 0x9c

	.section	.drectve,"yn"
	.ascii	" /DEFAULTLIB:libcmt.lib"
	.ascii	" /DEFAULTLIB:oldnames.lib"
	.section	.debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	241
	.long	.Ltmp3-.Ltmp2           # Subsection size
.Ltmp2:
	.short	.Ltmp5-.Ltmp4           # Record length
.Ltmp4:
	.short	4412                    # Record kind: S_COMPILE3
	.long	1                       # Flags and language
	.short	208                     # CPUType
	.short	8                       # Frontend version
	.short	0
	.short	0
	.short	0
	.short	8000                    # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 8.0.0 "  # Null-terminated compiler version string
.Ltmp5:
.Ltmp3:
	.p2align	2
	.long	241                     # Symbol subsection for main
	.long	.Ltmp7-.Ltmp6           # Subsection size
.Ltmp6:
	.short	.Ltmp9-.Ltmp8           # Record length
.Ltmp8:
	.short	4423                    # Record kind: S_GPROC32_ID
	.long	0                       # PtrParent
	.long	0                       # PtrEnd
	.long	0                       # PtrNext
	.long	.Lfunc_end0-main        # Code size
	.long	0                       # Offset after prologue
	.long	0                       # Offset before epilogue
	.long	4099                    # Function type index
	.secrel32	main            # Function section relative address
	.secidx	main                    # Function section index
	.byte	0                       # Flags
	.asciz	"main"                  # Function name
.Ltmp9:
	.short	.Ltmp11-.Ltmp10         # Record length
.Ltmp10:
	.short	4114                    # Record kind: S_FRAMEPROC
	.long	24                      # FrameSize
	.long	0                       # Padding
	.long	0                       # Offset of padding
	.long	0                       # Bytes of callee saved registers
	.long	0                       # Exception handler offset
	.short	0                       # Exception handler section
	.long	81920                   # Flags (defines frame register)
.Ltmp11:
	.short	.Ltmp13-.Ltmp12         # Record length
.Ltmp12:
	.short	4414                    # Record kind: S_LOCAL
	.long	116                     # TypeIndex
	.short	1                       # Flags
	.asciz	"argc"
.Ltmp13:
	.cv_def_range	 .Ltmp0 .Ltmp1, "B\021\004\000\000\000"
	.short	.Ltmp15-.Ltmp14         # Record length
.Ltmp14:
	.short	4414                    # Record kind: S_LOCAL
	.long	4096                    # TypeIndex
	.short	1                       # Flags
	.asciz	"argv"
.Ltmp15:
	.cv_def_range	 .Ltmp0 .Ltmp1, "B\021\b\000\000\000"
	.short	2                       # Record length
	.short	4431                    # Record kind: S_PROC_ID_END
.Ltmp7:
	.p2align	2
	.cv_linetable	0, main, .Lfunc_end0
	.long	241                     # Symbol subsection for globals
	.long	.Ltmp17-.Ltmp16         # Subsection size
.Ltmp16:
	.short	.Ltmp19-.Ltmp18         # Record length
.Ltmp18:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4104                    # Type
	.short  0					    # Value
	.asciz	"GlobalLUEA"            # Name
.Ltmp19:
	.short	.Ltmp21-.Ltmp20         # Record length
.Ltmp20:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4104                    # Type
	.short  1000				    # Value
	.asciz	"GlobalLUEB"            # Name
.Ltmp21:
	.short	.Ltmp23-.Ltmp22         # Record length
.Ltmp22:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4104                    # Type
	.byte   0x00, 0x80, 0xf0		# Value
	.asciz	"GlobalLUEC"            # Name
.Ltmp23:
	.short	.Ltmp25-.Ltmp24         # Record length
.Ltmp24:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4108                    # Type
	.byte   0x00, 0x00				# Value
	.asciz	"GlobalLSEA"            # Name
.Ltmp25:
	.short	.Ltmp27-.Ltmp26         # Record length
.Ltmp26:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4108                    # Type
	.byte   0x0A, 0x80, 0xD8, 0xFC  # Value
	.byte   0xFF, 0xFF, 0xFF, 0xFF
	.byte   0xFF, 0x7F
	.asciz	"GlobalLSEB"            # Name
.Ltmp27:
	.short	.Ltmp29-.Ltmp28         # Record length
.Ltmp28:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4108                    # Type
	.byte   0x09, 0x80, 0x28, 0x03  # Value
	.byte   0x00, 0x00, 0x00, 0x00
	.byte   0x00, 0x80
	.asciz	"GlobalLSEC"            # Name
.Ltmp29:
	.short	.Ltmp31-.Ltmp30         # Record length
.Ltmp30:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4112                    # Type
	.byte   0x00, 0x00              # Value
	.asciz	"GlobalUEA"             # Name
.Ltmp31:
	.short	.Ltmp33-.Ltmp32         # Record length
.Ltmp32:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4112                    # Type
	.byte   0xe8, 0x03              # Value
	.asciz	"GlobalUEB"             # Name
.Ltmp33:
	.short	.Ltmp35-.Ltmp34         # Record length
.Ltmp34:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4112                    # Type
	.byte   0x04, 0x80, 0x80, 0x3d  # Value
	.byte   0xf1, 0xff
	.asciz	"GlobalUEC"             # Name
.Ltmp35:
	.short	.Ltmp37-.Ltmp36         # Record length
.Ltmp36:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4116                    # Type
	.byte   0x00, 0x00              # Value
	.asciz	"GlobalSEA"             # Name
.Ltmp37:
	.short	.Ltmp39-.Ltmp38         # Record length
.Ltmp38:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4116                    # Type
	.byte   0x04, 0x80, 0xc0, 0x9e  # Value
	.byte   0xf8, 0x7f
	.asciz	"GlobalSEB"             # Name
.Ltmp39:
	.short	.Ltmp41-.Ltmp40         # Record length
.Ltmp40:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4116                    # Type
	.byte   0x03, 0x80, 0x40, 0x61  # Value
	.byte   0x07, 0x80
	.asciz	"GlobalSEC"             # Name
.Ltmp41:
	.short	.Ltmp43-.Ltmp42         # Record length
.Ltmp42:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4120                    # Type
	.byte   0x00, 0x00              # Value
	.asciz	"GlobalSUEA"            # Name
.Ltmp43:
	.short	.Ltmp45-.Ltmp44         # Record length
.Ltmp44:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4120                    # Type
	.byte   0x64, 0x00              # Value
	.asciz	"GlobalSUEB"            # Name
.Ltmp45:
	.short	.Ltmp47-.Ltmp46         # Record length
.Ltmp46:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4120                    # Type
	.byte   0xc8, 0x00              # Value
	.asciz	"GlobalSUEC"            # Name
.Ltmp47:
	.short	.Ltmp49-.Ltmp48         # Record length
.Ltmp48:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4124                    # Type
	.byte   0x00, 0x00              # Value
	.asciz	"GlobalSSEA"            # Name
.Ltmp49:
	.short	.Ltmp51-.Ltmp50         # Record length
.Ltmp50:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4124                    # Type
	.byte   0x64, 0x00              # Value
	.asciz	"GlobalSSEB"            # Name
.Ltmp51:
	.short	.Ltmp53-.Ltmp52         # Record length
.Ltmp52:
	.short	4359                    # Record kind: S_CONSTANT
	.long	4124                    # Type
	.byte   0x00, 0x80, 0x9c        # Value
	.asciz	"GlobalSSEC"            # Name
.Ltmp53:
.Ltmp17:
	.p2align	2
	.cv_filechecksums               # File index to string table offset subsection
	.cv_stringtable                 # String table
	.long	241
	.long	.Ltmp55-.Ltmp54         # Subsection size
.Ltmp54:
	.short	6                       # Record length
	.short	4428                    # Record kind: S_BUILDINFO
	.long	4127                    # LF_BUILDINFO index
.Ltmp55:
	.p2align	2
	.section	.debug$T,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	# Pointer (0x1000) {
	#   TypeLeafKind: LF_POINTER (0x1002)
	#   PointeeType: char* (0x670)
	#   PointerAttributes: 0x1000C
	#   PtrType: Near64 (0xC)
	#   PtrMode: Pointer (0x0)
	#   IsFlat: 0
	#   IsConst: 0
	#   IsVolatile: 0
	#   IsUnaligned: 0
	#   IsRestrict: 0
	#   SizeOf: 8
	# }
	.byte	0x0a, 0x00, 0x02, 0x10
	.byte	0x70, 0x06, 0x00, 0x00
	.byte	0x0c, 0x00, 0x01, 0x00
	# ArgList (0x1001) {
	#   TypeLeafKind: LF_ARGLIST (0x1201)
	#   NumArgs: 2
	#   Arguments [
	#     ArgType: int (0x74)
	#     ArgType: char** (0x1000)
	#   ]
	# }
	.byte	0x0e, 0x00, 0x01, 0x12
	.byte	0x02, 0x00, 0x00, 0x00
	.byte	0x74, 0x00, 0x00, 0x00
	.byte	0x00, 0x10, 0x00, 0x00
	# Procedure (0x1002) {
	#   TypeLeafKind: LF_PROCEDURE (0x1008)
	#   ReturnType: int (0x74)
	#   CallingConvention: NearC (0x0)
	#   FunctionOptions [ (0x0)
	#   ]
	#   NumParameters: 2
	#   ArgListType: (int, char**) (0x1001)
	# }
	.byte	0x0e, 0x00, 0x08, 0x10
	.byte	0x74, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x02, 0x00
	.byte	0x01, 0x10, 0x00, 0x00
	# FuncId (0x1003) {
	#   TypeLeafKind: LF_FUNC_ID (0x1601)
	#   ParentScope: 0x0
	#   FunctionType: int (int, char**) (0x1002)
	#   Name: main
	# }
	.byte	0x12, 0x00, 0x01, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x02, 0x10, 0x00, 0x00
	.byte	0x6d, 0x61, 0x69, 0x6e
	.byte	0x00, 0xf3, 0xf2, 0xf1
	# FieldList (0x1004) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: LUE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 1000
	#     Name: LUE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 18446744073709551600
	#     Name: LUE_C
	#   }
	# }
	.byte	0x2e, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x4c, 0x55
	.byte	0x45, 0x5f, 0x41, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0xe8, 0x03, 0x4c, 0x55
	.byte	0x45, 0x5f, 0x42, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x0a, 0x80, 0xf0, 0xff
	.byte	0xff, 0xff, 0xff, 0xff
	.byte	0xff, 0xff, 0x4c, 0x55
	.byte	0x45, 0x5f, 0x43, 0x00
	# Enum (0x1005) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: unsigned __int64 (0x23)
	#   FieldListType: <field list> (0x1004)
	#   Name: A::B::C::LargeUnsignedEnum
	#   LinkageName: .?AW4LargeUnsignedEnum@C@B@A@@
	# }
	.byte	0x4a, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x23, 0x00, 0x00, 0x00
	.byte	0x04, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x4c, 0x61, 0x72
	.byte	0x67, 0x65, 0x55, 0x6e
	.byte	0x73, 0x69, 0x67, 0x6e
	.byte	0x65, 0x64, 0x45, 0x6e
	.byte	0x75, 0x6d, 0x00, 0x2e
	.byte	0x3f, 0x41, 0x57, 0x34
	.byte	0x4c, 0x61, 0x72, 0x67
	.byte	0x65, 0x55, 0x6e, 0x73
	.byte	0x69, 0x67, 0x6e, 0x65
	.byte	0x64, 0x45, 0x6e, 0x75
	.byte	0x6d, 0x40, 0x43, 0x40
	.byte	0x42, 0x40, 0x41, 0x40
	.byte	0x40, 0x00, 0xf2, 0xf1
	# StringId (0x1006) {
	#   TypeLeafKind: LF_STRING_ID (0x1605)
	#   Id: 0x0
	#   StringData: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp
	# }
	.byte	0x46, 0x00, 0x05, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x44, 0x3a, 0x5c, 0x73
	.byte	0x72, 0x63, 0x5c, 0x6c
	.byte	0x6c, 0x76, 0x6d, 0x2d
	.byte	0x6d, 0x6f, 0x6e, 0x6f
	.byte	0x5c, 0x6c, 0x6c, 0x64
	.byte	0x62, 0x5c, 0x6c, 0x69
	.byte	0x74, 0x5c, 0x53, 0x79
	.byte	0x6d, 0x62, 0x6f, 0x6c
	.byte	0x46, 0x69, 0x6c, 0x65
	.byte	0x5c, 0x4e, 0x61, 0x74
	.byte	0x69, 0x76, 0x65, 0x50
	.byte	0x44, 0x42, 0x5c, 0x73
	.byte	0x5f, 0x63, 0x6f, 0x6e
	.byte	0x73, 0x74, 0x61, 0x6e
	.byte	0x74, 0x2e, 0x63, 0x70
	.byte	0x70, 0x00, 0xf2, 0xf1
	# UdtSourceLine (0x1007) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::LargeUnsignedEnum (0x1005)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 14
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x05, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x0e, 0x00, 0x00, 0x00
	# Modifier (0x1008) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::LargeUnsignedEnum (0x1005)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x05, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# FieldList (0x1009) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: LSE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 9223372036854775000
	#     Name: LSE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 9223372036854776616
	#     Name: LSE_C
	#   }
	# }
	.byte	0x36, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x4c, 0x53
	.byte	0x45, 0x5f, 0x41, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x0a, 0x80, 0xd8, 0xfc
	.byte	0xff, 0xff, 0xff, 0xff
	.byte	0xff, 0x7f, 0x4c, 0x53
	.byte	0x45, 0x5f, 0x42, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x0a, 0x80, 0x28, 0x03
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x80, 0x4c, 0x53
	.byte	0x45, 0x5f, 0x43, 0x00
	# Enum (0x100A) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: __int64 (0x13)
	#   FieldListType: <field list> (0x1009)
	#   Name: A::B::C::LargeSignedEnum
	#   LinkageName: .?AW4LargeSignedEnum@C@B@A@@
	# }
	.byte	0x46, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x13, 0x00, 0x00, 0x00
	.byte	0x09, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x4c, 0x61, 0x72
	.byte	0x67, 0x65, 0x53, 0x69
	.byte	0x67, 0x6e, 0x65, 0x64
	.byte	0x45, 0x6e, 0x75, 0x6d
	.byte	0x00, 0x2e, 0x3f, 0x41
	.byte	0x57, 0x34, 0x4c, 0x61
	.byte	0x72, 0x67, 0x65, 0x53
	.byte	0x69, 0x67, 0x6e, 0x65
	.byte	0x64, 0x45, 0x6e, 0x75
	.byte	0x6d, 0x40, 0x43, 0x40
	.byte	0x42, 0x40, 0x41, 0x40
	.byte	0x40, 0x00, 0xf2, 0xf1
	# UdtSourceLine (0x100B) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::LargeSignedEnum (0x100A)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 20
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x0a, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x14, 0x00, 0x00, 0x00
	# Modifier (0x100C) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::LargeSignedEnum (0x100A)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x0a, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# FieldList (0x100D) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: UE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 1000
	#     Name: UE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 4294000000
	#     Name: UE_C
	#   }
	# }
	.byte	0x2a, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x55, 0x45
	.byte	0x5f, 0x41, 0x00, 0xf1
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0xe8, 0x03, 0x55, 0x45
	.byte	0x5f, 0x42, 0x00, 0xf1
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x04, 0x80, 0x80, 0x3d
	.byte	0xf1, 0xff, 0x55, 0x45
	.byte	0x5f, 0x43, 0x00, 0xf1
	# Enum (0x100E) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: unsigned (0x75)
	#   FieldListType: <field list> (0x100D)
	#   Name: A::B::C::UnsignedEnum
	#   LinkageName: .?AW4UnsignedEnum@C@B@A@@
	# }
	.byte	0x3e, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x75, 0x00, 0x00, 0x00
	.byte	0x0d, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x55, 0x6e, 0x73
	.byte	0x69, 0x67, 0x6e, 0x65
	.byte	0x64, 0x45, 0x6e, 0x75
	.byte	0x6d, 0x00, 0x2e, 0x3f
	.byte	0x41, 0x57, 0x34, 0x55
	.byte	0x6e, 0x73, 0x69, 0x67
	.byte	0x6e, 0x65, 0x64, 0x45
	.byte	0x6e, 0x75, 0x6d, 0x40
	.byte	0x43, 0x40, 0x42, 0x40
	.byte	0x41, 0x40, 0x40, 0x00
	# UdtSourceLine (0x100F) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::UnsignedEnum (0x100E)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 26
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x0e, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x1a, 0x00, 0x00, 0x00
	# Modifier (0x1010) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::UnsignedEnum (0x100E)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x0e, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# FieldList (0x1011) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: SE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 2147000000
	#     Name: SE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 18446744071562551616
	#     Name: SE_C
	#   }
	# }
	.byte	0x32, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x53, 0x45
	.byte	0x5f, 0x41, 0x00, 0xf1
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x04, 0x80, 0xc0, 0x9e
	.byte	0xf8, 0x7f, 0x53, 0x45
	.byte	0x5f, 0x42, 0x00, 0xf1
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x0a, 0x80, 0x40, 0x61
	.byte	0x07, 0x80, 0xff, 0xff
	.byte	0xff, 0xff, 0x53, 0x45
	.byte	0x5f, 0x43, 0x00, 0xf1
	# Enum (0x1012) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: int (0x74)
	#   FieldListType: <field list> (0x1011)
	#   Name: A::B::C::SignedEnum
	#   LinkageName: .?AW4SignedEnum@C@B@A@@
	# }
	.byte	0x3a, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x74, 0x00, 0x00, 0x00
	.byte	0x11, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x53, 0x69, 0x67
	.byte	0x6e, 0x65, 0x64, 0x45
	.byte	0x6e, 0x75, 0x6d, 0x00
	.byte	0x2e, 0x3f, 0x41, 0x57
	.byte	0x34, 0x53, 0x69, 0x67
	.byte	0x6e, 0x65, 0x64, 0x45
	.byte	0x6e, 0x75, 0x6d, 0x40
	.byte	0x43, 0x40, 0x42, 0x40
	.byte	0x41, 0x40, 0x40, 0x00
	# UdtSourceLine (0x1013) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::SignedEnum (0x1012)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 32
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x12, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x20, 0x00, 0x00, 0x00
	# Modifier (0x1014) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::SignedEnum (0x1012)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x12, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# FieldList (0x1015) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: SUE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 100
	#     Name: SUE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 200
	#     Name: SUE_C
	#   }
	# }
	.byte	0x26, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x53, 0x55
	.byte	0x45, 0x5f, 0x41, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x64, 0x00, 0x53, 0x55
	.byte	0x45, 0x5f, 0x42, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0xc8, 0x00, 0x53, 0x55
	.byte	0x45, 0x5f, 0x43, 0x00
	# Enum (0x1016) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: unsigned char (0x20)
	#   FieldListType: <field list> (0x1015)
	#   Name: A::B::C::SmallUnsignedEnum
	#   LinkageName: .?AW4SmallUnsignedEnum@C@B@A@@
	# }
	.byte	0x4a, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x20, 0x00, 0x00, 0x00
	.byte	0x15, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x53, 0x6d, 0x61
	.byte	0x6c, 0x6c, 0x55, 0x6e
	.byte	0x73, 0x69, 0x67, 0x6e
	.byte	0x65, 0x64, 0x45, 0x6e
	.byte	0x75, 0x6d, 0x00, 0x2e
	.byte	0x3f, 0x41, 0x57, 0x34
	.byte	0x53, 0x6d, 0x61, 0x6c
	.byte	0x6c, 0x55, 0x6e, 0x73
	.byte	0x69, 0x67, 0x6e, 0x65
	.byte	0x64, 0x45, 0x6e, 0x75
	.byte	0x6d, 0x40, 0x43, 0x40
	.byte	0x42, 0x40, 0x41, 0x40
	.byte	0x40, 0x00, 0xf2, 0xf1
	# UdtSourceLine (0x1017) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::SmallUnsignedEnum (0x1016)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 38
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x16, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x26, 0x00, 0x00, 0x00
	# Modifier (0x1018) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::SmallUnsignedEnum (0x1016)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x16, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# FieldList (0x1019) {
	#   TypeLeafKind: LF_FIELDLIST (0x1203)
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 0
	#     Name: SSE_A
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 100
	#     Name: SSE_B
	#   }
	#   Enumerator {
	#     TypeLeafKind: LF_ENUMERATE (0x1502)
	#     AccessSpecifier: Public (0x3)
	#     EnumValue: 18446744073709551516
	#     Name: SSE_C
	#   }
	# }
	.byte	0x2e, 0x00, 0x03, 0x12
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x00, 0x00, 0x53, 0x53
	.byte	0x45, 0x5f, 0x41, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x64, 0x00, 0x53, 0x53
	.byte	0x45, 0x5f, 0x42, 0x00
	.byte	0x02, 0x15, 0x03, 0x00
	.byte	0x0a, 0x80, 0x9c, 0xff
	.byte	0xff, 0xff, 0xff, 0xff
	.byte	0xff, 0xff, 0x53, 0x53
	.byte	0x45, 0x5f, 0x43, 0x00
	# Enum (0x101A) {
	#   TypeLeafKind: LF_ENUM (0x1507)
	#   NumEnumerators: 3
	#   Properties [ (0x200)
	#     HasUniqueName (0x200)
	#   ]
	#   UnderlyingType: char (0x70)
	#   FieldListType: <field list> (0x1019)
	#   Name: A::B::C::SmallSignedEnum
	#   LinkageName: .?AW4SmallSignedEnum@C@B@A@@
	# }
	.byte	0x46, 0x00, 0x07, 0x15
	.byte	0x03, 0x00, 0x00, 0x02
	.byte	0x70, 0x00, 0x00, 0x00
	.byte	0x19, 0x10, 0x00, 0x00
	.byte	0x41, 0x3a, 0x3a, 0x42
	.byte	0x3a, 0x3a, 0x43, 0x3a
	.byte	0x3a, 0x53, 0x6d, 0x61
	.byte	0x6c, 0x6c, 0x53, 0x69
	.byte	0x67, 0x6e, 0x65, 0x64
	.byte	0x45, 0x6e, 0x75, 0x6d
	.byte	0x00, 0x2e, 0x3f, 0x41
	.byte	0x57, 0x34, 0x53, 0x6d
	.byte	0x61, 0x6c, 0x6c, 0x53
	.byte	0x69, 0x67, 0x6e, 0x65
	.byte	0x64, 0x45, 0x6e, 0x75
	.byte	0x6d, 0x40, 0x43, 0x40
	.byte	0x42, 0x40, 0x41, 0x40
	.byte	0x40, 0x00, 0xf2, 0xf1
	# UdtSourceLine (0x101B) {
	#   TypeLeafKind: LF_UDT_SRC_LINE (0x1606)
	#   UDT: A::B::C::SmallSignedEnum (0x101A)
	#   SourceFile: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\s_constant.cpp (0x1006)
	#   LineNumber: 44
	# }
	.byte	0x0e, 0x00, 0x06, 0x16
	.byte	0x1a, 0x10, 0x00, 0x00
	.byte	0x06, 0x10, 0x00, 0x00
	.byte	0x2c, 0x00, 0x00, 0x00
	# Modifier (0x101C) {
	#   TypeLeafKind: LF_MODIFIER (0x1001)
	#   ModifiedType: A::B::C::SmallSignedEnum (0x101A)
	#   Modifiers [ (0x1)
	#     Const (0x1)
	#   ]
	# }
	.byte	0x0a, 0x00, 0x01, 0x10
	.byte	0x1a, 0x10, 0x00, 0x00
	.byte	0x01, 0x00, 0xf2, 0xf1
	# StringId (0x101D) {
	#   TypeLeafKind: LF_STRING_ID (0x1605)
	#   Id: 0x0
	#   StringData: D:\\src\\llvmbuild\\ninja-x64
	# }
	.byte	0x26, 0x00, 0x05, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x44, 0x3a, 0x5c, 0x5c
	.byte	0x73, 0x72, 0x63, 0x5c
	.byte	0x5c, 0x6c, 0x6c, 0x76
	.byte	0x6d, 0x62, 0x75, 0x69
	.byte	0x6c, 0x64, 0x5c, 0x5c
	.byte	0x6e, 0x69, 0x6e, 0x6a
	.byte	0x61, 0x2d, 0x78, 0x36
	.byte	0x34, 0x00, 0xf2, 0xf1
	# StringId (0x101E) {
	#   TypeLeafKind: LF_STRING_ID (0x1605)
	#   Id: 0x0
	#   StringData: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\variable.cpp
	# }
	.byte	0x42, 0x00, 0x05, 0x16
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x44, 0x3a, 0x5c, 0x73
	.byte	0x72, 0x63, 0x5c, 0x6c
	.byte	0x6c, 0x76, 0x6d, 0x2d
	.byte	0x6d, 0x6f, 0x6e, 0x6f
	.byte	0x5c, 0x6c, 0x6c, 0x64
	.byte	0x62, 0x5c, 0x6c, 0x69
	.byte	0x74, 0x5c, 0x53, 0x79
	.byte	0x6d, 0x62, 0x6f, 0x6c
	.byte	0x46, 0x69, 0x6c, 0x65
	.byte	0x5c, 0x4e, 0x61, 0x74
	.byte	0x69, 0x76, 0x65, 0x50
	.byte	0x44, 0x42, 0x5c, 0x76
	.byte	0x61, 0x72, 0x69, 0x61
	.byte	0x62, 0x6c, 0x65, 0x2e
	.byte	0x63, 0x70, 0x70, 0x00
	# BuildInfo (0x101F) {
	#   TypeLeafKind: LF_BUILDINFO (0x1603)
	#   NumArgs: 5
	#   Arguments [
	#     ArgType: D:\\src\\llvmbuild\\ninja-x64 (0x101D)
	#     ArgType: 0x0
	#     ArgType: D:\src\llvm-mono\lldb\lit\SymbolFile\NativePDB\variable.cpp (0x101E)
	#     ArgType: 0x0
	#     ArgType: 0x0
	#   ]
	# }
	.byte	0x1a, 0x00, 0x03, 0x16
	.byte	0x05, 0x00, 0x1d, 0x10
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x1e, 0x10
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0x00, 0x00
	.byte	0x00, 0x00, 0xf2, 0xf1

	.addrsig
	.addrsig_sym GlobalLUEA
	.addrsig_sym GlobalLUEB
	.addrsig_sym GlobalLUEC
	.addrsig_sym GlobalLSEA
	.addrsig_sym GlobalLSEB
	.addrsig_sym GlobalLSEC
	.addrsig_sym GlobalUEA
	.addrsig_sym GlobalUEB
	.addrsig_sym GlobalUEC
	.addrsig_sym GlobalSEA
	.addrsig_sym GlobalSEB
	.addrsig_sym GlobalSEC
	.addrsig_sym GlobalSUEA
	.addrsig_sym GlobalSUEB
	.addrsig_sym GlobalSUEC
	.addrsig_sym GlobalSSEA
	.addrsig_sym GlobalSSEB
	.addrsig_sym GlobalSSEC
