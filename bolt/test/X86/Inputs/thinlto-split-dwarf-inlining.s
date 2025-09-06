## This file was built through distrubuted Thinlto in order to inline callee.cpp and callee2.cpp into main.cpp
## clang++ -O3 -g -gdwarf-5 -gsplit-dwarf -fsplit-dwarf-inlining -flto=thin -c main.cpp -o main.thinlto
## clang++ -O3 -g -gdwarf-5 -gsplit-dwarf -fsplit-dwarf-inlining -flto=thin -c callee.cpp -o callee.thinlto
## clang++ -O3 -g -gdwarf-5 -gsplit-dwarf -fsplit-dwarf-inlining -flto=thin -c callee2.cpp -o callee2.thinlto
## clang++ -fuse-ld=lld -flto=thin -Wl,-plugin-opt,thinlto-index-only=main.thinlto.index main.thinlto callee.thinlto callee2.thinlto -o index
## clang++ -O3 -g -gdwarf-5 -gsplit-dwarf -fsplit-dwarf-inlining -fthinlto-index=main.thinlto.thinlto.bc -c -x ir main.thinlto -S -o main.S

## main.cpp
## int hotFunction(int x);
## int hotFunction2(int x);
## int main(int argc, char **argv) {
##     int sum = 0;
##     for (int i = 0; i < 50000000; ++i) {
##         sum += hotFunction(i);
##         sum += hotFunction2(i);
##     }
##     if (sum)
##         return 0;
##     else
##         return 1;
## }
## callee.cpp
## int hotFunction(int x) {
##     if ((x & 1) == 0) {
##         x = x * 3 + 1;
##         if (x % 5 == 0) {
##             x += 7;
##         }
##     } else {
##         x = x * x;
##         if (x % 3 == 0) {
##             x -= 4;
##         }
##     }
##     return x;
## }
## callee2.cpp
## int hotFunction2(int x) {
##     if ((x & 2) == 0) {
##         x = x * 3 + 1;
##         if (x % 5 == 0) {
##             x += 7;
##         }
##     } else {
##         x = x * x;
##         if (x % 3 == 0) {
##             x -= 4;
##         }
##     }
##     return x;
## }

	.text
	.file	"main.cpp"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          # -- Begin function main
.LCPI0_0:
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	2                               # 0x2
	.long	3                               # 0x3
.LCPI0_1:
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
	.long	4                               # 0x4
.LCPI0_2:
	.long	1                               # 0x1
	.long	1                               # 0x1
	.long	1                               # 0x1
	.long	1                               # 0x1
.LCPI0_3:
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
	.long	2863311531                      # 0xaaaaaaab
.LCPI0_4:
	.long	2147483648                      # 0x80000000
	.long	2147483648                      # 0x80000000
	.long	2147483648                      # 0x80000000
	.long	2147483648                      # 0x80000000
.LCPI0_5:
	.long	3579139413                      # 0xd5555555
	.long	3579139413                      # 0xd5555555
	.long	3579139413                      # 0xd5555555
	.long	3579139413                      # 0xd5555555
.LCPI0_6:
	.long	4294967292                      # 0xfffffffc
	.long	4294967292                      # 0xfffffffc
	.long	4294967292                      # 0xfffffffc
	.long	4294967292                      # 0xfffffffc
.LCPI0_7:
	.long	3435973837                      # 0xcccccccd
	.long	3435973837                      # 0xcccccccd
	.long	3435973837                      # 0xcccccccd
	.long	3435973837                      # 0xcccccccd
.LCPI0_8:
	.long	3006477107                      # 0xb3333333
	.long	3006477107                      # 0xb3333333
	.long	3006477107                      # 0xb3333333
	.long	3006477107                      # 0xb3333333
.LCPI0_9:
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
	.long	8                               # 0x8
.LCPI0_10:
	.long	2                               # 0x2
	.long	2                               # 0x2
	.long	2                               # 0x2
	.long	2                               # 0x2
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin0:
	.file	1 "./" "main.cpp" md5 0x3f802f4e24573ca71e028348cd028728
	.loc	1 5 0                           # main.cpp:5:0
	.cfi_startproc
# %bb.0:                                # %vector.ph
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	movdqa	.LCPI0_0(%rip), %xmm3           # xmm3 = [0,1,2,3]
	movl	$50000000, %eax                 # imm = 0x2FAF080
.Ltmp0:
	#DEBUG_VALUE: main:sum <- 0
	#DEBUG_VALUE: i <- 0
	movdqa	.LCPI0_4(%rip), %xmm5           # xmm5 = [2147483648,2147483648,2147483648,2147483648]
	movdqa	.LCPI0_7(%rip), %xmm9           # xmm9 = [3435973837,3435973837,3435973837,3435973837]
	pxor	%xmm14, %xmm14
	pxor	%xmm12, %xmm12
	movdqa	.LCPI0_8(%rip), %xmm11          # xmm11 = [3006477107,3006477107,3006477107,3006477107]
.Ltmp1:
	.p2align	4, 0x90
.LBB0_1:                                # %vector.body
                                        # =>This Inner Loop Header: Depth=1
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: main:sum <- 0
	#DEBUG_VALUE: i <- 0
	movdqa	%xmm3, %xmm1
	paddd	.LCPI0_1(%rip), %xmm1
.Ltmp2:
	.file	2 "./" "callee.cpp" md5 0xaf5c6b5b41909e1a2265d9ef8d74e9d0
	.loc	2 9 15 prologue_end             # callee.cpp:9:15
	movdqa	%xmm3, %xmm2
	pmuludq	%xmm3, %xmm2
	pshufd	$232, %xmm2, %xmm15             # xmm15 = xmm2[0,2,2,3]
	pshufd	$245, %xmm3, %xmm13             # xmm13 = xmm3[1,1,3,3]
	pmuludq	%xmm13, %xmm13
	pshufd	$232, %xmm13, %xmm0             # xmm0 = xmm13[0,2,2,3]
	punpckldq	%xmm0, %xmm15           # xmm15 = xmm15[0],xmm0[0],xmm15[1],xmm0[1]
	pshufd	$245, %xmm1, %xmm0              # xmm0 = xmm1[1,1,3,3]
.Ltmp3:
	.loc	2 4 15                          # callee.cpp:4:15
	movdqa	%xmm1, %xmm10
	paddd	%xmm1, %xmm10
	paddd	%xmm1, %xmm10
.Ltmp4:
	.loc	2 9 15                          # callee.cpp:9:15
	pmuludq	%xmm1, %xmm1
	pshufd	$232, %xmm1, %xmm7              # xmm7 = xmm1[0,2,2,3]
	pmuludq	%xmm0, %xmm0
	pshufd	$232, %xmm0, %xmm8              # xmm8 = xmm0[0,2,2,3]
	punpckldq	%xmm8, %xmm7            # xmm7 = xmm7[0],xmm8[0],xmm7[1],xmm8[1]
	movdqa	.LCPI0_3(%rip), %xmm4           # xmm4 = [2863311531,2863311531,2863311531,2863311531]
.Ltmp5:
	.loc	2 10 19                         # callee.cpp:10:19
	pmuludq	%xmm4, %xmm2
	pshufd	$232, %xmm2, %xmm2              # xmm2 = xmm2[0,2,2,3]
	pmuludq	%xmm4, %xmm13
	pshufd	$232, %xmm13, %xmm8             # xmm8 = xmm13[0,2,2,3]
	punpckldq	%xmm8, %xmm2            # xmm2 = xmm2[0],xmm8[0],xmm2[1],xmm8[1]
	pxor	%xmm5, %xmm2
	movdqa	.LCPI0_5(%rip), %xmm6           # xmm6 = [3579139413,3579139413,3579139413,3579139413]
	pcmpgtd	%xmm6, %xmm2
	pmuludq	%xmm4, %xmm1
	pshufd	$232, %xmm1, %xmm1              # xmm1 = xmm1[0,2,2,3]
	pmuludq	%xmm4, %xmm0
	pshufd	$232, %xmm0, %xmm0              # xmm0 = xmm0[0,2,2,3]
	punpckldq	%xmm0, %xmm1            # xmm1 = xmm1[0],xmm0[0],xmm1[1],xmm0[1]
	pxor	%xmm5, %xmm1
	pcmpgtd	%xmm6, %xmm1
.Ltmp6:
	.loc	2 10 13 is_stmt 0               # callee.cpp:10:13
	movdqa	%xmm2, %xmm0
	pand	%xmm15, %xmm0
	movdqa	.LCPI0_6(%rip), %xmm6           # xmm6 = [4294967292,4294967292,4294967292,4294967292]
	paddd	%xmm6, %xmm15
	pandn	%xmm15, %xmm2
	por	%xmm0, %xmm2
	movdqa	%xmm1, %xmm0
	pand	%xmm7, %xmm0
	paddd	%xmm6, %xmm7
	pandn	%xmm7, %xmm1
	por	%xmm0, %xmm1
.Ltmp7:
	.loc	2 4 15 is_stmt 1                # callee.cpp:4:15
	movdqa	%xmm3, %xmm7
	paddd	%xmm3, %xmm7
	paddd	%xmm3, %xmm7
	.loc	2 4 19 is_stmt 0                # callee.cpp:4:19
	movdqa	%xmm7, %xmm8
	pcmpeqd	%xmm0, %xmm0
	psubd	%xmm0, %xmm8
	movdqa	%xmm10, %xmm6
	psubd	%xmm0, %xmm6
.Ltmp8:
	.loc	2 5 19 is_stmt 1                # callee.cpp:5:19
	movdqa	%xmm8, %xmm0
	pmuludq	%xmm9, %xmm0
	pshufd	$232, %xmm0, %xmm13             # xmm13 = xmm0[0,2,2,3]
	pshufd	$245, %xmm8, %xmm0              # xmm0 = xmm8[1,1,3,3]
	pmuludq	%xmm9, %xmm0
	pshufd	$232, %xmm0, %xmm0              # xmm0 = xmm0[0,2,2,3]
	punpckldq	%xmm0, %xmm13           # xmm13 = xmm13[0],xmm0[0],xmm13[1],xmm0[1]
	movdqa	%xmm6, %xmm0
	pmuludq	%xmm9, %xmm0
	pshufd	$232, %xmm0, %xmm0              # xmm0 = xmm0[0,2,2,3]
	pshufd	$245, %xmm6, %xmm15             # xmm15 = xmm6[1,1,3,3]
	pmuludq	%xmm9, %xmm15
	pshufd	$232, %xmm15, %xmm15            # xmm15 = xmm15[0,2,2,3]
	punpckldq	%xmm15, %xmm0           # xmm0 = xmm0[0],xmm15[0],xmm0[1],xmm15[1]
	pxor	%xmm5, %xmm13
	pcmpgtd	%xmm11, %xmm13
	movdqa	.LCPI0_9(%rip), %xmm4           # xmm4 = [8,8,8,8]
	paddd	%xmm4, %xmm7
.Ltmp9:
	.loc	2 5 13 is_stmt 0                # callee.cpp:5:13
	pand	%xmm13, %xmm8
	pandn	%xmm7, %xmm13
	por	%xmm8, %xmm13
.Ltmp10:
	.loc	2 3 12 is_stmt 1                # callee.cpp:3:12
	movdqa	%xmm3, %xmm15
	pand	.LCPI0_2(%rip), %xmm15
	pxor	%xmm8, %xmm8
	.loc	2 3 17 is_stmt 0                # callee.cpp:3:17
	pcmpeqd	%xmm8, %xmm15
.Ltmp11:
	.loc	2 5 19 is_stmt 1                # callee.cpp:5:19
	pxor	%xmm5, %xmm0
	pcmpgtd	%xmm11, %xmm0
	paddd	%xmm4, %xmm10
.Ltmp12:
	.loc	2 5 13 is_stmt 0                # callee.cpp:5:13
	pand	%xmm0, %xmm6
	pandn	%xmm10, %xmm0
	por	%xmm6, %xmm0
.Ltmp13:
	.loc	2 0 0                           # callee.cpp:0:0
	movdqa	%xmm15, %xmm6
	pandn	%xmm2, %xmm6
	movdqa	%xmm13, %xmm7
	pand	%xmm15, %xmm7
	por	%xmm6, %xmm7
.Ltmp14:
	.loc	1 8 13 is_stmt 1                # main.cpp:8:13
	paddd	%xmm14, %xmm7
.Ltmp15:
	.loc	2 0 0 is_stmt 0                 # callee.cpp:0:0
	movdqa	%xmm0, %xmm6
	pand	%xmm15, %xmm6
	pandn	%xmm1, %xmm15
	por	%xmm6, %xmm15
.Ltmp16:
	.loc	1 8 13                          # main.cpp:8:13
	paddd	%xmm12, %xmm15
.Ltmp17:
	.file	3 "./" "callee2.cpp" md5 0xdc3a45c8e5ee5cd983e9eb6e4bfdd449
	.loc	3 3 12 is_stmt 1                # callee2.cpp:3:12
	movdqa	%xmm3, %xmm12
	pand	.LCPI0_10(%rip), %xmm12
	.loc	3 3 17 is_stmt 0                # callee2.cpp:3:17
	pcmpeqd	%xmm8, %xmm12
	.loc	3 0 0                           # callee2.cpp:0:0
	movdqa	%xmm12, %xmm6
	pandn	%xmm2, %xmm6
	pand	%xmm12, %xmm13
	por	%xmm6, %xmm13
.Ltmp18:
	.loc	1 9 13 is_stmt 1                # main.cpp:9:13
	movdqa	%xmm13, %xmm14
	paddd	%xmm7, %xmm14
.Ltmp19:
	.loc	3 0 0 is_stmt 0                 # callee2.cpp:0:0
	pand	%xmm12, %xmm0
	pandn	%xmm1, %xmm12
	por	%xmm0, %xmm12
.Ltmp20:
	.loc	1 9 13                          # main.cpp:9:13
	paddd	%xmm15, %xmm12
	paddd	%xmm4, %xmm3
.Ltmp21:
	.loc	1 7 35 is_stmt 1                # main.cpp:7:35
	addl	$-8, %eax
	jne	.LBB0_1
.Ltmp22:
# %bb.2:                                # %middle.block
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: main:sum <- 0
	#DEBUG_VALUE: i <- 0
	.loc	1 7 5 is_stmt 0                 # main.cpp:7:5
	paddd	%xmm14, %xmm12
	pshufd	$238, %xmm12, %xmm0             # xmm0 = xmm12[2,3,2,3]
	paddd	%xmm12, %xmm0
	pshufd	$85, %xmm0, %xmm1               # xmm1 = xmm0[1,1,1,1]
	paddd	%xmm0, %xmm1
	movd	%xmm1, %ecx
.Ltmp23:
	.loc	1 11 9 is_stmt 1                # main.cpp:11:9
	xorl	%eax, %eax
	testl	%ecx, %ecx
	sete	%al
.Ltmp24:
	.loc	1 15 1                          # main.cpp:15:1
	retq
.Ltmp25:
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	116                             # DW_AT_rnglists_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	16                              # DW_FORM_ref_addr
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	74                              # DW_TAG_skeleton_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	114                             # DW_AT_str_offsets_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	37                              # DW_FORM_strx1
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	115                             # DW_AT_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	8954242594720362350
	.byte	1                               # Abbrev [1] 0x14:0x33 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	4                               # DW_AT_dwo_name
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.long	.Lrnglists_table_base0          # DW_AT_rnglists_base
	.byte	2                               # Abbrev [2] 0x2c:0x1a DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	3                               # DW_AT_name
	.byte	3                               # Abbrev [3] 0x33:0x9 DW_TAG_inlined_subroutine
	.long	.debug_info+109                 # DW_AT_abstract_origin
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	8                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	3                               # Abbrev [3] 0x3c:0x9 DW_TAG_inlined_subroutine
	.long	.debug_info+150                 # DW_AT_abstract_origin
	.byte	1                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	9                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
.Ldebug_info_end0:
.Lcu_begin1:
	.long	.Ldebug_info_end1-.Ldebug_info_start1 # Length of Unit
.Ldebug_info_start1:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	0
	.byte	4                               # Abbrev [4] 0x14:0x15 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	5                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	6                               # DW_AT_name
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	5                               # Abbrev [5] 0x26:0x2 DW_TAG_subprogram
	.byte	1                               # DW_AT_name
                                        # DW_AT_inline
	.byte	0                               # End Of Children Mark
.Ldebug_info_end1:
.Lcu_begin2:
	.long	.Ldebug_info_end2-.Ldebug_info_start2 # Length of Unit
.Ldebug_info_start2:
	.short	5                               # DWARF version number
	.byte	4                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.quad	0
	.byte	4                               # Abbrev [4] 0x14:0x15 DW_TAG_skeleton_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lstr_offsets_base0             # DW_AT_str_offsets_base
	.byte	0                               # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.byte	5                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	7                               # DW_AT_name
	.long	.Laddr_table_base0              # DW_AT_addr_base
	.byte	5                               # Abbrev [5] 0x26:0x2 DW_TAG_subprogram
	.byte	2                               # DW_AT_name
                                        # DW_AT_inline
	.byte	0                               # End Of Children Mark
.Ldebug_info_end2:
	.section	.debug_rnglists,"",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	2                               # Offset entry count
.Lrnglists_table_base0:
	.long	.Ldebug_ranges2-.Lrnglists_table_base0
	.long	.Ldebug_ranges3-.Lrnglists_table_base0
.Ldebug_ranges2:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp14-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp15-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp16-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_ranges3:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp17-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp18-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp19-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp20-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	36                              # Length of String Offsets Set
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"./" # string offset=0
.Lskel_string1:
	.asciz	"hotFunction"                   # string offset=45
.Lskel_string2:
	.asciz	"hotFunction2"                  # string offset=57
.Lskel_string3:
	.asciz	"main"                          # string offset=70
.Lskel_string4:
	.asciz	"main.exe-thinlto-split-dwarf-inlining.dwo"                      # string offset=75
.Lskel_string5:
	.asciz	"clang version 16.0.6 (gitlab@git.byted.org:sys/llvm-project.git ae0ef36eb8428ef48cb345a0e0c9ad6b3f289590)" # string offset=84
.Lskel_string6:
	.asciz	"callee.cpp"                    # string offset=190
.Lskel_string7:
	.asciz	"callee2.cpp"                   # string offset=201
	.section	.debug_str_offsets,"",@progbits
	.long	.Lskel_string0
	.long	.Lskel_string1
	.long	.Lskel_string2
	.long	.Lskel_string3
	.long	.Lskel_string4
	.long	.Lskel_string5
	.long	.Lskel_string6
	.long	.Lskel_string7
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	64                              # Length of String Offsets Set
	.short	5
	.short	0
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z11hotFunctioni"              # string offset=0
.Linfo_string1:
	.asciz	"hotFunction"                   # string offset=17
.Linfo_string2:
	.asciz	"int"                           # string offset=29
.Linfo_string3:
	.asciz	"x"                             # string offset=33
.Linfo_string4:
	.asciz	"_Z12hotFunction2i"             # string offset=35
.Linfo_string5:
	.asciz	"hotFunction2"                  # string offset=53
.Linfo_string6:
	.asciz	"main"                          # string offset=66
.Linfo_string7:
	.asciz	"argc"                          # string offset=71
.Linfo_string8:
	.asciz	"argv"                          # string offset=76
.Linfo_string9:
	.asciz	"char"                          # string offset=81
.Linfo_string10:
	.asciz	"sum"                           # string offset=86
.Linfo_string11:
	.asciz	"i"                             # string offset=90
.Linfo_string12:
	.asciz	"clang version 16.0.6 (gitlab@git.byted.org:sys/llvm-project.git ae0ef36eb8428ef48cb345a0e0c9ad6b3f289590)" # string offset=92
.Linfo_string13:
	.asciz	"main.cpp"                      # string offset=198
.Linfo_string14:
	.asciz	"main.exe-thinlto-split-dwarf-inlining.dwo"                      # string offset=207
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	17
	.long	29
	.long	33
	.long	35
	.long	53
	.long	66
	.long	71
	.long	76
	.long	81
	.long	86
	.long	90
	.long	92
	.long	198
	.long	207
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	8954242594720362350
	.byte	1                               # Abbrev [1] 0x14:0x8c DW_TAG_compile_unit
	.byte	12                              # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	13                              # DW_AT_name
	.byte	14                              # DW_AT_dwo_name
	.byte	2                               # Abbrev [2] 0x1a:0x12 DW_TAG_subprogram
	.byte	0                               # DW_AT_linkage_name
	.byte	1                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x23:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	2                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	4                               # Abbrev [4] 0x2c:0x4 DW_TAG_base_type
	.byte	2                               # DW_AT_name
	.byte	5                               # DW_AT_encoding
	.byte	4                               # DW_AT_byte_size
	.byte	2                               # Abbrev [2] 0x30:0x12 DW_TAG_subprogram
	.byte	4                               # DW_AT_linkage_name
	.byte	5                               # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_inline
	.byte	3                               # Abbrev [3] 0x39:0x8 DW_TAG_formal_parameter
	.byte	3                               # DW_AT_name
	.byte	3                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	0                               # End Of Children Mark
	.byte	5                               # Abbrev [5] 0x42:0x4f DW_TAG_subprogram
	.byte	0                               # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.byte	1                               # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	6                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
                                        # DW_AT_external
	.byte	6                               # Abbrev [6] 0x51:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	85
	.byte	7                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	6                               # Abbrev [6] 0x5b:0xa DW_TAG_formal_parameter
	.byte	1                               # DW_AT_location
	.byte	84
	.byte	8                               # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	5                               # DW_AT_decl_line
	.long	145                             # DW_AT_type
	.byte	7                               # Abbrev [7] 0x65:0x9 DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.byte	10                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	6                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	8                               # Abbrev [8] 0x6e:0x22 DW_TAG_lexical_block
	.byte	1                               # DW_AT_low_pc
	.long	.Ltmp23-.Ltmp2                  # DW_AT_high_pc
	.byte	7                               # Abbrev [7] 0x74:0x9 DW_TAG_variable
	.byte	0                               # DW_AT_const_value
	.byte	11                              # DW_AT_name
	.byte	1                               # DW_AT_decl_file
	.byte	7                               # DW_AT_decl_line
	.long	44                              # DW_AT_type
	.byte	9                               # Abbrev [9] 0x7d:0x9 DW_TAG_inlined_subroutine
	.long	26                              # DW_AT_abstract_origin
	.byte	0                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	8                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	9                               # Abbrev [9] 0x86:0x9 DW_TAG_inlined_subroutine
	.long	48                              # DW_AT_abstract_origin
	.byte	1                               # DW_AT_ranges
	.byte	1                               # DW_AT_call_file
	.byte	9                               # DW_AT_call_line
	.byte	16                              # DW_AT_call_column
	.byte	0                               # End Of Children Mark
	.byte	0                               # End Of Children Mark
	.byte	10                              # Abbrev [10] 0x91:0x5 DW_TAG_pointer_type
	.long	150                             # DW_AT_type
	.byte	10                              # Abbrev [10] 0x96:0x5 DW_TAG_pointer_type
	.long	155                             # DW_AT_type
	.byte	4                               # Abbrev [4] 0x9b:0x4 DW_TAG_base_type
	.byte	9                               # DW_AT_name
	.byte	6                               # DW_AT_encoding
	.byte	1                               # DW_AT_byte_size
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.byte	37                              # DW_FORM_strx1
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	118                             # DW_AT_dwo_name
	.byte	37                              # DW_FORM_strx1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	110                             # DW_AT_linkage_name
	.byte	37                              # DW_FORM_strx1
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	32                              # DW_AT_inline
	.byte	33                              # DW_FORM_implicit_const
	.byte	1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	36                              # DW_TAG_base_type
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	62                              # DW_AT_encoding
	.byte	11                              # DW_FORM_data1
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	46                              # DW_TAG_subprogram
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	64                              # DW_AT_frame_base
	.byte	24                              # DW_FORM_exprloc
	.byte	122                             # DW_AT_call_all_calls
	.byte	25                              # DW_FORM_flag_present
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	6                               # Abbreviation Code
	.byte	5                               # DW_TAG_formal_parameter
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	7                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	28                              # DW_AT_const_value
	.byte	13                              # DW_FORM_sdata
	.byte	3                               # DW_AT_name
	.byte	37                              # DW_FORM_strx1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	8                               # Abbreviation Code
	.byte	11                              # DW_TAG_lexical_block
	.byte	1                               # DW_CHILDREN_yes
	.byte	17                              # DW_AT_low_pc
	.byte	27                              # DW_FORM_addrx
	.byte	18                              # DW_AT_high_pc
	.byte	6                               # DW_FORM_data4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	9                               # Abbreviation Code
	.byte	29                              # DW_TAG_inlined_subroutine
	.byte	0                               # DW_CHILDREN_no
	.byte	49                              # DW_AT_abstract_origin
	.byte	19                              # DW_FORM_ref4
	.byte	85                              # DW_AT_ranges
	.byte	35                              # DW_FORM_rnglistx
	.byte	88                              # DW_AT_call_file
	.byte	11                              # DW_FORM_data1
	.byte	89                              # DW_AT_call_line
	.byte	11                              # DW_FORM_data1
	.byte	87                              # DW_AT_call_column
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	10                              # Abbreviation Code
	.byte	15                              # DW_TAG_pointer_type
	.byte	0                               # DW_CHILDREN_no
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_rnglists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 # Length
.Ldebug_list_header_start1:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	2                               # Offset entry count
.Lrnglists_dwo_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_dwo_table_base0
	.long	.Ldebug_ranges1-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp2-.Lfunc_begin0           #   starting offset
	.uleb128 .Ltmp14-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp15-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp16-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_ranges1:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp17-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp18-.Lfunc_begin0          #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 .Ltmp19-.Lfunc_begin0          #   starting offset
	.uleb128 .Ltmp20-.Lfunc_begin0          #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end1:
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                               # DWARF version number
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Ltmp2
.Ldebug_addr_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	71                              # Compilation Unit Length
	.long	26                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"hotFunction"                   # External Name
	.long	48                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"hotFunction2"                  # External Name
	.long	66                              # DIE offset
	.byte	48                              # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                          # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	71                              # Compilation Unit Length
	.long	44                              # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"int"                           # External Name
	.long	155                             # DIE offset
	.byte	144                             # Attributes: TYPE, STATIC
	.asciz	"char"                          # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end1-.LpubNames_start1 # Length of Public Names Info
.LpubNames_start1:
	.short	2                               # DWARF Version
	.long	.Lcu_begin1                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubNames_end1:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end1-.LpubTypes_start1 # Length of Public Types Info
.LpubTypes_start1:
	.short	2                               # DWARF Version
	.long	.Lcu_begin1                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end1:
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end2-.LpubNames_start2 # Length of Public Names Info
.LpubNames_start2:
	.short	2                               # DWARF Version
	.long	.Lcu_begin2                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubNames_end2:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end2-.LpubTypes_start2 # Length of Public Types Info
.LpubTypes_start2:
	.short	2                               # DWARF Version
	.long	.Lcu_begin2                     # Offset of Compilation Unit Info
	.long	41                              # Compilation Unit Length
	.long	0                               # End Mark
.LpubTypes_end2:
	.ident	"clang version 16.0.6 (gitlab@git.byted.org:sys/llvm-project.git ae0ef36eb8428ef48cb345a0e0c9ad6b3f289590)"
	.ident	"clang version 16.0.6 (gitlab@git.byted.org:sys/llvm-project.git ae0ef36eb8428ef48cb345a0e0c9ad6b3f289590)"
	.ident	"clang version 16.0.6 (gitlab@git.byted.org:sys/llvm-project.git ae0ef36eb8428ef48cb345a0e0c9ad6b3f289590)"
	.section	.GCC.command.line,"MS",@progbits,1
	.zero	1
	.ascii	"/data00/tiger/cpp_tools/x86_64_x86_64_clang_1606/bin/clang-16 --driver-mode=g++ --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -pipe --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -O3 -g -gdwarf-5 -gsplit-dwarf -fno-debug-info-for-profiling -fsplit-dwarf-inlining -flto=thin -c main.cpp -o main.thinlto -Werror=return-type -Wno-error=return-type-c-linkage"
	.zero	1
	.ascii	"/data00/tiger/cpp_tools/x86_64_x86_64_clang_1606/bin/clang-16 --driver-mode=g++ --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -pipe --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -O3 -g -gdwarf-5 -gsplit-dwarf -fno-debug-info-for-profiling -fsplit-dwarf-inlining -flto=thin -c callee.cpp -o callee.thinlto -Werror=return-type -Wno-error=return-type-c-linkage"
	.zero	1
	.ascii	"/data00/tiger/cpp_tools/x86_64_x86_64_clang_1606/bin/clang-16 --driver-mode=g++ --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -pipe --gcc-toolchain=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606 --sysroot=/opt/tiger/cpp_tools/x86_64_x86_64_clang_1606/sysroot -O3 -g -gdwarf-5 -gsplit-dwarf -fno-debug-info-for-profiling -fsplit-dwarf-inlining -flto=thin -c callee2.cpp -o callee2.thinlto -Werror=return-type -Wno-error=return-type-c-linkage"
	.zero	1
	.section	.debug_gnu_pubtypes,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
