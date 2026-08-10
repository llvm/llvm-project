# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %p/Inputs/ppc64-got-to-pcrel-relaxation-def.s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so --soname=t2
# RUN: ld.lld %t1.o %t2.o -o %t
# RUN: ld.lld %t1.o %t2.so -o %ts
# RUN: ld.lld %t1.o %t2.o -o %tn --no-pcrel-optimize
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=CHECK-S
# RUN: llvm-objdump -d --no-show-raw-insn %ts | FileCheck %s --check-prefix=CHECK-D
# RUN: llvm-objdump -d --no-show-raw-insn %tn | FileCheck %s --check-prefix=CHECK-D

# RUN: llvm-mc -filetype=obj -triple=powerpc64 %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64 %p/Inputs/ppc64-got-to-pcrel-relaxation-def.s -o %t2.o
# RUN: ld.lld --shared %t2.o -o %t2.so --soname=t2
# RUN: ld.lld %t1.o %t2.o -o %t
# RUN: ld.lld %t1.o %t2.so -o %ts
# RUN: ld.lld %t1.o %t2.o -o %tn --no-pcrel-optimize
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=CHECK-S
# RUN: llvm-objdump -d --no-show-raw-insn %ts | FileCheck %s --check-prefix=CHECK-D
# RUN: llvm-objdump -d --no-show-raw-insn %tn | FileCheck %s --check-prefix=CHECK-D

# CHECK-S-LABEL: <check_LBZ_STB>:
# CHECK-S-NEXT:    plbz 10
# CHECK-S-NEXT:    paddi 9
# CHECK-S-NEXT:    li 3, 0
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    rldicl 9, 9, 9, 60
# CHECK-S-NEXT:    add 9, 9, 10
# CHECK-S-NEXT:    pstb 9
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LBZ_STB>:
# CHECK-D-NEXT:    pld 8
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    li 3, 0
# CHECK-D-NEXT:    lbz 10, 0(8)
# CHECK-D-NEXT:    rldicl 9, 9, 9, 60
# CHECK-D-NEXT:    add 9, 9, 10
# CHECK-D-NEXT:    pld 10
# CHECK-D-NEXT:    stb 9, 0(10)
# CHECK-D-NEXT:    blr
check_LBZ_STB:
  pld 8,useVal@got@pcrel(0),1
.Lpcrel1:
  pld 9,useAddr@got@pcrel(0),1
  li 3,0
  .reloc .Lpcrel1-8,R_PPC64_PCREL_OPT,.-(.Lpcrel1-8)
  lbz 10,0(8)
  rldicl 9,9,9,60
  add 9,9,10
  pld 10,storeVal@got@pcrel(0),1
.Lpcrel2:
  .reloc .Lpcrel2-8,R_PPC64_PCREL_OPT,.-(.Lpcrel2-8)
  stb 9,0(10)
  blr

# CHECK-S-LABEL: <check_LHZ_STH>:
# CHECK-S-NEXT:    plhz 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    psth 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LHZ_STH>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lhz 3, 0(9)
# CHECK-D-NEXT:    nop
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    sth 3, 0(9)
# CHECK-D-NEXT:    blr
check_LHZ_STH:
  pld 9,useVal_ushort@got@pcrel(0),1
.Lpcrel3:
  .reloc .Lpcrel3-8,R_PPC64_PCREL_OPT,.-(.Lpcrel3-8)
  lhz 3,0(9)
  pld 9,storeVal_ushort@got@pcrel(0),1
.Lpcrel4:
  .reloc .Lpcrel4-8,R_PPC64_PCREL_OPT,.-(.Lpcrel4-8)
  sth 3,0(9)
  blr

# CHECK-S-LABEL: <check_LWZ_STW>:
# CHECK-S-NEXT:    plwz 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstw 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LWZ_STW>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lwz 3, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stw 3, 0(9)
# CHECK-D-NEXT:    blr
check_LWZ_STW:
  pld 9,useVal_uint@got@pcrel(0),1
.Lpcrel5:
  .reloc .Lpcrel5-8,R_PPC64_PCREL_OPT,.-(.Lpcrel5-8)
  lwz 3,0(9)
  pld 9,storeVal_uint@got@pcrel(0),1
.Lpcrel6:
  .reloc .Lpcrel6-8,R_PPC64_PCREL_OPT,.-(.Lpcrel6-8)
  stw 3,0(9)
  blr

# CHECK-S-LABEL: <check_LFS_STFS>:
# CHECK-S-NEXT:    plfs 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstfs 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LFS_STFS>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lfs 1, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stfs 1, 0(9)
# CHECK-D-NEXT:    blr
check_LFS_STFS:
  pld 9,useVal_float@got@pcrel(0),1
.Lpcrel7:
  .reloc .Lpcrel7-8,R_PPC64_PCREL_OPT,.-(.Lpcrel7-8)
  lfs 1,0(9)
  pld 9,storeVal_float@got@pcrel(0),1
.Lpcrel8:
  .reloc .Lpcrel8-8,R_PPC64_PCREL_OPT,.-(.Lpcrel8-8)
  stfs 1,0(9)
  blr

# CHECK-S-LABEL: <check_LFD_STFD>:
# CHECK-S-NEXT:    plfd 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstfd 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LFD_STFD>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lfd 1, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stfd 1, 0(9)
# CHECK-D-NEXT:    blr
check_LFD_STFD:
  pld 9,useVal_double@got@pcrel(0),1
.Lpcrel9:
  .reloc .Lpcrel9-8,R_PPC64_PCREL_OPT,.-(.Lpcrel9-8)
  lfd 1,0(9)
  pld 9,storeVal_double@got@pcrel(0),1
.Lpcrel10:
  .reloc .Lpcrel10-8,R_PPC64_PCREL_OPT,.-(.Lpcrel10-8)
  stfd 1,0(9)
  blr

# CHECK-S-LABEL: <check_LWA_STW>:
# CHECK-S-NEXT:    mr 9, 3
# CHECK-S-NEXT:    plwa 3
# CHECK-S-NEXT:    pstw 9
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LWA_STW>:
# CHECK-D-NEXT:    mr 9, 3
# CHECK-D-NEXT:    pld 8
# CHECK-D-NEXT:    pld 10
# CHECK-D-NEXT:    lwa 3, 0(8)
# CHECK-D-NEXT:    stw 9, 0(10)
# CHECK-D-NEXT:    blr
check_LWA_STW:
  mr 9,3
  pld 8,useVal_sint@got@pcrel(0),1
.Lpcrel11:
  pld 10,storeVal_sint@got@pcrel(0),1
.Lpcrel12:
  .reloc .Lpcrel11-8,R_PPC64_PCREL_OPT,.-(.Lpcrel11-8)
  lwa 3,0(8)
  .reloc .Lpcrel12-8,R_PPC64_PCREL_OPT,.-(.Lpcrel12-8)
  stw 9,0(10)
  blr

# CHECK-S-LABEL: <check_LHA_STH>:
# CHECK-S-NEXT:    mr 9, 3
# CHECK-S-NEXT:    plha 3
# CHECK-S-NEXT:    psth 9
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LHA_STH>:
# CHECK-D-NEXT:    mr 9, 3
# CHECK-D-NEXT:    pld 8
# CHECK-D-NEXT:    pld 10
# CHECK-D-NEXT:    lha 3, 0(8)
# CHECK-D-NEXT:    sth 9, 0(10)
# CHECK-D-NEXT:    blr
check_LHA_STH:
  mr 9,3
  pld 8,useVal_sshort@got@pcrel(0),1
.Lpcrel13:
  pld 10,storeVal_sshort@got@pcrel(0),1
.Lpcrel14:
  .reloc .Lpcrel13-8,R_PPC64_PCREL_OPT,.-(.Lpcrel13-8)
  lha 3,0(8)
  .reloc .Lpcrel14-8,R_PPC64_PCREL_OPT,.-(.Lpcrel14-8)
  sth 9,0(10)
  blr

# CHECK-S-LABEL: <check_LD_STD>:
# CHECK-S-NEXT:    pld 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstd 3
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LD_STD>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    ld 3, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    std 3, 0(9)
# CHECK-D-NEXT:    blr
check_LD_STD:
  pld 9,useVal_longlong@got@pcrel(0),1
.Lpcrel15:
  .reloc .Lpcrel15-8,R_PPC64_PCREL_OPT,.-(.Lpcrel15-8)
  ld 3,0(9)
  pld 9,storeVal_longlong@got@pcrel(0),1
.Lpcrel16:
  .reloc .Lpcrel16-8,R_PPC64_PCREL_OPT,.-(.Lpcrel16-8)
  std 3,0(9)
  blr

# CHECK-S-LABEL: <check_LXV_STXV>:
# CHECK-S-NEXT:    plxv 34
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstxv 34
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LXV_STXV>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lxv 34, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stxv 34, 0(9)
# CHECK-D-NEXT:    blr
check_LXV_STXV:
  pld 9,useVal_vector@got@pcrel(0),1
.Lpcrel17:
  .reloc .Lpcrel17-8,R_PPC64_PCREL_OPT,.-(.Lpcrel17-8)
  lxv 34,0(9)
  pld 9,storeVal_vector@got@pcrel(0),1
.Lpcrel18:
  .reloc .Lpcrel18-8,R_PPC64_PCREL_OPT,.-(.Lpcrel18-8)
  stxv 34,0(9)
  blr

# CHECK-S-LABEL: <check_LXSSP_STXSSP>:
# CHECK-S-NEXT:    plxssp 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstxssp 1
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LXSSP_STXSSP>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lxssp 1, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stxssp 1, 0(9)
# CHECK-D-NEXT:    blr
check_LXSSP_STXSSP:
  pld 9,useVal_float@got@pcrel(0),1
.Lpcrel19:
  .reloc .Lpcrel19-8,R_PPC64_PCREL_OPT,.-(.Lpcrel19-8)
  lxssp 1,0(9)
  pld 9,storeVal_float@got@pcrel(0),1
.Lpcrel20:
  .reloc .Lpcrel20-8,R_PPC64_PCREL_OPT,.-(.Lpcrel20-8)
  stxssp 1,0(9)
  blr

# CHECK-S-LABEL: <check_LXSD_STXSD>:
# CHECK-S-NEXT:    plxsd 1, [[#ADDR1:]]
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstxsd 1, [[#ADDR2:]]
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LXSD_STXSD>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lxsd 1, 0(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stxsd 1, 0(9)
# CHECK-D-NEXT:    blr
check_LXSD_STXSD:
  pld 9,useVal_double@got@pcrel(0),1
.Lpcrel21:
  .reloc .Lpcrel21-8,R_PPC64_PCREL_OPT,.-(.Lpcrel21-8)
  lxsd 1,0(9)
  pld 9,storeVal_double@got@pcrel(0),1
.Lpcrel22:
  .reloc .Lpcrel22-8,R_PPC64_PCREL_OPT,.-(.Lpcrel22-8)
  stxsd 1,0(9)
  blr

# The respective displacements are computed relative to the PC which advanced
# by 28 bytes in this function. Since the displacements in the two access
# instructions are 8 and 32 so the displacements are those computed above minus
# 20 and plus 4 (+8 - 28 and +32 - 28) respectively.
# CHECK-S-LABEL: <check_LXSD_STXSD_aggr>:
# CHECK-S-NEXT:    plxsd 1, [[#ADDR1-20]]
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    pstxsd 1, [[#ADDR2+4]]
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LXSD_STXSD_aggr>:
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    lxsd 1, 8(9)
# CHECK-D-NEXT:    pld 9
# CHECK-D-NEXT:    stxsd 1, 32(9)
# CHECK-D-NEXT:    blr
check_LXSD_STXSD_aggr:
  pld 9,useVal_double@got@pcrel(0),1
.Lpcrel23:
  .reloc .Lpcrel23-8,R_PPC64_PCREL_OPT,.-(.Lpcrel23-8)
  lxsd 1,8(9)
  pld 9,storeVal_double@got@pcrel(0),1
.Lpcrel24:
  .reloc .Lpcrel24-8,R_PPC64_PCREL_OPT,.-(.Lpcrel24-8)
  stxsd 1,32(9)
  blr

# This includes a nop but that is not emitted by the linker.
# It is an alignment nop to prevent the prefixed instruction from
# crossing a 64-byte boundary.
# CHECK-S-LABEL: <check_LD_STD_W_PADDI>:
# CHECK-S-NEXT:    paddi 9
# CHECK-S-NEXT:    ld 3, 0(9)
# CHECK-S-NEXT:    nop
# CHECK-S-NEXT:    paddi 9
# CHECK-S-NEXT:    std 3, 0(9)
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LD_STD_W_PADDI>:
# CHECK-D-NEXT:    paddi 9
# CHECK-D-NEXT:    ld 3, 0(9)
# CHECK-D-NEXT:    nop
# CHECK-D-NEXT:    paddi 9
# CHECK-D-NEXT:    std 3, 0(9)
# CHECK-D-NEXT:    blr
check_LD_STD_W_PADDI:
  paddi 9,0,useVal_longlong@got@pcrel,1
.Lpcrel25:
  .reloc .Lpcrel25-8,R_PPC64_PCREL_OPT,.-(.Lpcrel25-8)
  ld 3,0(9)
  paddi 9,0,storeVal_longlong@got@pcrel,1
.Lpcrel26:
  .reloc .Lpcrel26-8,R_PPC64_PCREL_OPT,.-(.Lpcrel26-8)
  std 3,0(9)
  blr
# CHECK-S-LABEL: <check_LXSD_STXSD_aggr_notoc>:
# CHECK-S-NEXT:    paddi 3, 0, -12, 1
# CHECK-S-NEXT:    lwz 4, 8(3)
# CHECK-S-NEXT:    paddi 3, 0, -24, 1
# CHECK-S-NEXT:    stw 4, 32(3)
# CHECK-S-NEXT:    blr

# CHECK-D-LABEL: <check_LXSD_STXSD_aggr_notoc>:
# CHECK-D-NEXT:    paddi 3, 0, -12, 1
# CHECK-D-NEXT:    lwz 4, 8(3)
# CHECK-D-NEXT:    paddi 3, 0, -24, 1
# CHECK-D-NEXT:    stw 4, 32(3)
# CHECK-D-NEXT:    blr
.type	Arr,@object                     # @Arr
.globl	Arr
.p2align	2
Arr:
.long	11                              # 0xb
.long	22                              # 0x16
.long	33                              # 0x21
check_LXSD_STXSD_aggr_notoc:
  paddi 3, 0, Arr@PCREL, 1
.Lpcrel27:
  .reloc .Lpcrel27-8,R_PPC64_PCREL_OPT,.-(.Lpcrel27-8)
  lwz 4,8(3)
  paddi 3, 0, Arr@PCREL, 1
.Lpcrel28:
  .reloc .Lpcrel28-8,R_PPC64_PCREL_OPT,.-(.Lpcrel28-8)
  stw 4,32(3)
  blr

