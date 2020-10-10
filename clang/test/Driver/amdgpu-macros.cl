// Check that appropriate macros are defined for every supported AMDGPU
// "-target" and "-mcpu" options.

//
// R600-based processors.
//

// RUN: %clang -E -dM -target r600 -mcpu=r600 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv630 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv635 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R600 %s
// RUN: %clang -E -dM -target r600 -mcpu=r630 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,R630 %s
// RUN: %clang -E -dM -target r600 -mcpu=rs780 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s
// RUN: %clang -E -dM -target r600 -mcpu=rs880 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv610 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv620 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RS880 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv670 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV670 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv710 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV710 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv730 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV730 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv740 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV770 %s
// RUN: %clang -E -dM -target r600 -mcpu=rv770 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,RV770 %s
// RUN: %clang -E -dM -target r600 -mcpu=cedar %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CEDAR %s
// RUN: %clang -E -dM -target r600 -mcpu=palm %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CEDAR %s
// RUN: %clang -E -dM -target r600 -mcpu=cypress %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CYPRESS %s
// RUN: %clang -E -dM -target r600 -mcpu=hemlock %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CYPRESS %s
// RUN: %clang -E -dM -target r600 -mcpu=juniper %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,JUNIPER %s
// RUN: %clang -E -dM -target r600 -mcpu=redwood %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,REDWOOD %s
// RUN: %clang -E -dM -target r600 -mcpu=sumo %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,SUMO %s
// RUN: %clang -E -dM -target r600 -mcpu=sumo2 %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,SUMO %s
// RUN: %clang -E -dM -target r600 -mcpu=barts %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,BARTS %s
// RUN: %clang -E -dM -target r600 -mcpu=caicos %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAICOS %s
// RUN: %clang -E -dM -target r600 -mcpu=aruba %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s
// RUN: %clang -E -dM -target r600 -mcpu=cayman %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s
// RUN: %clang -E -dM -target r600 -mcpu=turks %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,TURKS %s

// R600-NOT:    #define FP_FAST_FMA 1
// R630-NOT:    #define FP_FAST_FMA 1
// RS880-NOT:   #define FP_FAST_FMA 1
// RV670-NOT:   #define FP_FAST_FMA 1
// RV710-NOT:   #define FP_FAST_FMA 1
// RV730-NOT:   #define FP_FAST_FMA 1
// RV770-NOT:   #define FP_FAST_FMA 1
// CEDAR-NOT:   #define FP_FAST_FMA 1
// CYPRESS-NOT: #define FP_FAST_FMA 1
// JUNIPER-NOT: #define FP_FAST_FMA 1
// REDWOOD-NOT: #define FP_FAST_FMA 1
// SUMO-NOT:    #define FP_FAST_FMA 1
// BARTS-NOT:   #define FP_FAST_FMA 1
// CAICOS-NOT:  #define FP_FAST_FMA 1
// CAYMAN-NOT:  #define FP_FAST_FMA 1
// TURKS-NOT:   #define FP_FAST_FMA 1

// R600-NOT:    #define FP_FAST_FMAF 1
// R630-NOT:    #define FP_FAST_FMAF 1
// RS880-NOT:   #define FP_FAST_FMAF 1
// RV670-NOT:   #define FP_FAST_FMAF 1
// RV710-NOT:   #define FP_FAST_FMAF 1
// RV730-NOT:   #define FP_FAST_FMAF 1
// RV770-NOT:   #define FP_FAST_FMAF 1
// CEDAR-NOT:   #define FP_FAST_FMAF 1
// CYPRESS-NOT: #define FP_FAST_FMAF 1
// JUNIPER-NOT: #define FP_FAST_FMAF 1
// REDWOOD-NOT: #define FP_FAST_FMAF 1
// SUMO-NOT:    #define FP_FAST_FMAF 1
// BARTS-NOT:   #define FP_FAST_FMAF 1
// CAICOS-NOT:  #define FP_FAST_FMAF 1
// CAYMAN-NOT:  #define FP_FAST_FMAF 1
// TURKS-NOT:   #define FP_FAST_FMAF 1

// ARCH-R600-DAG: #define __AMDGPU__ 1
// ARCH-R600-DAG: #define __AMD__ 1

// R600-NOT:    #define __HAS_FMAF__ 1
// R630-NOT:    #define __HAS_FMAF__ 1
// RS880-NOT:   #define __HAS_FMAF__ 1
// RV670-NOT:   #define __HAS_FMAF__ 1
// RV710-NOT:   #define __HAS_FMAF__ 1
// RV730-NOT:   #define __HAS_FMAF__ 1
// RV770-NOT:   #define __HAS_FMAF__ 1
// CEDAR-NOT:   #define __HAS_FMAF__ 1
// CYPRESS-DAG: #define __HAS_FMAF__ 1
// JUNIPER-NOT: #define __HAS_FMAF__ 1
// REDWOOD-NOT: #define __HAS_FMAF__ 1
// SUMO-NOT:    #define __HAS_FMAF__ 1
// BARTS-NOT:   #define __HAS_FMAF__ 1
// CAICOS-NOT:  #define __HAS_FMAF__ 1
// CAYMAN-DAG:  #define __HAS_FMAF__ 1
// TURKS-NOT:   #define __HAS_FMAF__ 1

// R600-NOT:    #define __HAS_FP64__ 1
// R630-NOT:    #define __HAS_FP64__ 1
// RS880-NOT:   #define __HAS_FP64__ 1
// RV670-NOT:   #define __HAS_FP64__ 1
// RV710-NOT:   #define __HAS_FP64__ 1
// RV730-NOT:   #define __HAS_FP64__ 1
// RV770-NOT:   #define __HAS_FP64__ 1
// CEDAR-NOT:   #define __HAS_FP64__ 1
// CYPRESS-NOT: #define __HAS_FP64__ 1
// JUNIPER-NOT: #define __HAS_FP64__ 1
// REDWOOD-NOT: #define __HAS_FP64__ 1
// SUMO-NOT:    #define __HAS_FP64__ 1
// BARTS-NOT:   #define __HAS_FP64__ 1
// CAICOS-NOT:  #define __HAS_FP64__ 1
// CAYMAN-NOT:  #define __HAS_FP64__ 1
// TURKS-NOT:   #define __HAS_FP64__ 1

// R600-NOT:    #define __HAS_LDEXPF__ 1
// R630-NOT:    #define __HAS_LDEXPF__ 1
// RS880-NOT:   #define __HAS_LDEXPF__ 1
// RV670-NOT:   #define __HAS_LDEXPF__ 1
// RV710-NOT:   #define __HAS_LDEXPF__ 1
// RV730-NOT:   #define __HAS_LDEXPF__ 1
// RV770-NOT:   #define __HAS_LDEXPF__ 1
// CEDAR-NOT:   #define __HAS_LDEXPF__ 1
// CYPRESS-NOT: #define __HAS_LDEXPF__ 1
// JUNIPER-NOT: #define __HAS_LDEXPF__ 1
// REDWOOD-NOT: #define __HAS_LDEXPF__ 1
// SUMO-NOT:    #define __HAS_LDEXPF__ 1
// BARTS-NOT:   #define __HAS_LDEXPF__ 1
// CAICOS-NOT:  #define __HAS_LDEXPF__ 1
// CAYMAN-NOT:  #define __HAS_LDEXPF__ 1
// TURKS-NOT:   #define __HAS_LDEXPF__ 1

// ARCH-R600-DAG: #define __R600__ 1

// R600-DAG:    #define __r600__ 1
// R630-DAG:    #define __r630__ 1
// RS880-DAG:   #define __rs880__ 1
// RV670-DAG:   #define __rv670__ 1
// RV710-DAG:   #define __rv710__ 1
// RV730-DAG:   #define __rv730__ 1
// RV770-DAG:   #define __rv770__ 1
// CEDAR-DAG:   #define __cedar__ 1
// CYPRESS-DAG: #define __cypress__ 1
// JUNIPER-DAG: #define __juniper__ 1
// REDWOOD-DAG: #define __redwood__ 1
// SUMO-DAG:    #define __sumo__ 1
// BARTS-DAG:   #define __barts__ 1
// CAICOS-DAG:  #define __caicos__ 1
// CAYMAN-DAG:  #define __cayman__ 1
// TURKS-DAG:   #define __turks__ 1

//
// AMDGCN-based processors.
//

// RUN: %clang -E -dM -target amdgcn -mcpu=gfx600 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX600 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX600 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx601 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=pitcairn %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=verde %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx602 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX602 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=hainan %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX602 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=oland %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX602 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx700 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX700 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=kaveri %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX700 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx701 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX701 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=hawaii %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX701 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx702 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX702 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx703 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX703 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=kabini %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX703 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=mullins %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX703 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx704 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX704 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=bonaire %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX704 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx705 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX705 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx801 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX801 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=carrizo %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX801 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=iceland %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=tonga %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris10 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris11 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx805 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX805 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=tongapro %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX805 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx810 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX810 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=stoney %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX810 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX900 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx902 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX902 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx904 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX904 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX906 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX908 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx909 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX909 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX1010 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1011 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX1011 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1012 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX1012 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1030 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX1030 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1031 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX1031 %s

// GFX600-DAG: #define FP_FAST_FMA 1
// GFX601-DAG: #define FP_FAST_FMA 1
// GFX602-DAG: #define FP_FAST_FMA 1
// GFX700-DAG: #define FP_FAST_FMA 1
// GFX701-DAG: #define FP_FAST_FMA 1
// GFX702-DAG: #define FP_FAST_FMA 1
// GFX703-DAG: #define FP_FAST_FMA 1
// GFX704-DAG: #define FP_FAST_FMA 1
// GFX705-DAG: #define FP_FAST_FMA 1
// GFX801-DAG: #define FP_FAST_FMA 1
// GFX802-DAG: #define FP_FAST_FMA 1
// GFX803-DAG: #define FP_FAST_FMA 1
// GFX805-DAG: #define FP_FAST_FMA 1
// GFX810-DAG: #define FP_FAST_FMA 1
// GFX900-DAG: #define FP_FAST_FMA 1
// GFX902-DAG: #define FP_FAST_FMA 1
// GFX904-DAG: #define FP_FAST_FMA 1
// GFX906-DAG: #define FP_FAST_FMA 1
// GFX908-DAG: #define FP_FAST_FMA 1
// GFX909-DAG: #define FP_FAST_FMA 1
// GFX1010-DAG: #define FP_FAST_FMA 1
// GFX1011-DAG: #define FP_FAST_FMA 1
// GFX1012-DAG: #define FP_FAST_FMA 1
// GFX1030-DAG: #define FP_FAST_FMA 1
// GFX1031-DAG: #define FP_FAST_FMA 1

// GFX600-DAG: #define FP_FAST_FMAF 1
// GFX601-NOT: #define FP_FAST_FMAF 1
// GFX602-NOT: #define FP_FAST_FMAF 1
// GFX700-NOT: #define FP_FAST_FMAF 1
// GFX701-DAG: #define FP_FAST_FMAF 1
// GFX702-DAG: #define FP_FAST_FMAF 1
// GFX703-NOT: #define FP_FAST_FMAF 1
// GFX704-NOT: #define FP_FAST_FMAF 1
// GFX705-NOT: #define FP_FAST_FMAF 1
// GFX801-DAG: #define FP_FAST_FMAF 1
// GFX802-NOT: #define FP_FAST_FMAF 1
// GFX803-NOT: #define FP_FAST_FMAF 1
// GFX805-NOT: #define FP_FAST_FMAF 1
// GFX810-NOT: #define FP_FAST_FMAF 1
// GFX900-DAG: #define FP_FAST_FMAF 1
// GFX902-DAG: #define FP_FAST_FMAF 1
// GFX904-DAG: #define FP_FAST_FMAF 1
// GFX906-DAG: #define FP_FAST_FMAF 1
// GFX908-DAG: #define FP_FAST_FMAF 1
// GFX909-DAG: #define FP_FAST_FMAF 1
// GFX1010-DAG: #define FP_FAST_FMAF 1
// GFX1011-DAG: #define FP_FAST_FMAF 1
// GFX1012-DAG: #define FP_FAST_FMAF 1
// GFX1030-DAG: #define FP_FAST_FMAF 1
// GFX1031-DAG: #define FP_FAST_FMAF 1

// ARCH-GCN-DAG: #define __AMDGCN__ 1
// ARCH-GCN-DAG: #define __AMDGPU__ 1
// ARCH-GCN-DAG: #define __AMD__ 1

// GFX600-DAG: #define __HAS_FMAF__ 1
// GFX601-DAG: #define __HAS_FMAF__ 1
// GFX602-DAG: #define __HAS_FMAF__ 1
// GFX700-DAG: #define __HAS_FMAF__ 1
// GFX701-DAG: #define __HAS_FMAF__ 1
// GFX702-DAG: #define __HAS_FMAF__ 1
// GFX703-DAG: #define __HAS_FMAF__ 1
// GFX704-DAG: #define __HAS_FMAF__ 1
// GFX705-DAG: #define __HAS_FMAF__ 1
// GFX801-DAG: #define __HAS_FMAF__ 1
// GFX802-DAG: #define __HAS_FMAF__ 1
// GFX803-DAG: #define __HAS_FMAF__ 1
// GFX805-DAG: #define __HAS_FMAF__ 1
// GFX810-DAG: #define __HAS_FMAF__ 1
// GFX900-DAG: #define __HAS_FMAF__ 1
// GFX902-DAG: #define __HAS_FMAF__ 1
// GFX904-DAG: #define __HAS_FMAF__ 1
// GFX906-DAG: #define __HAS_FMAF__ 1
// GFX908-DAG: #define __HAS_FMAF__ 1
// GFX909-DAG: #define __HAS_FMAF__ 1
// GFX1010-DAG: #define __HAS_FMAF__ 1
// GFX1011-DAG: #define __HAS_FMAF__ 1
// GFX1012-DAG: #define __HAS_FMAF__ 1
// GFX1030-DAG: #define __HAS_FMAF__ 1
// GFX1031-DAG: #define __HAS_FMAF__ 1

// GFX600-DAG: #define __HAS_FP64__ 1
// GFX601-DAG: #define __HAS_FP64__ 1
// GFX602-DAG: #define __HAS_FP64__ 1
// GFX700-DAG: #define __HAS_FP64__ 1
// GFX701-DAG: #define __HAS_FP64__ 1
// GFX702-DAG: #define __HAS_FP64__ 1
// GFX703-DAG: #define __HAS_FP64__ 1
// GFX704-DAG: #define __HAS_FP64__ 1
// GFX705-DAG: #define __HAS_FP64__ 1
// GFX801-DAG: #define __HAS_FP64__ 1
// GFX802-DAG: #define __HAS_FP64__ 1
// GFX803-DAG: #define __HAS_FP64__ 1
// GFX805-DAG: #define __HAS_FP64__ 1
// GFX810-DAG: #define __HAS_FP64__ 1
// GFX900-DAG: #define __HAS_FP64__ 1
// GFX902-DAG: #define __HAS_FP64__ 1
// GFX904-DAG: #define __HAS_FP64__ 1
// GFX906-DAG: #define __HAS_FP64__ 1
// GFX908-DAG: #define __HAS_FP64__ 1
// GFX909-DAG: #define __HAS_FP64__ 1
// GFX1010-DAG: #define __HAS_FP64__ 1
// GFX1011-DAG: #define __HAS_FP64__ 1
// GFX1012-DAG: #define __HAS_FP64__ 1
// GFX1030-DAG: #define __HAS_FP64__ 1
// GFX1031-DAG: #define __HAS_FP64__ 1

// GFX600-DAG: #define __HAS_LDEXPF__ 1
// GFX601-DAG: #define __HAS_LDEXPF__ 1
// GFX602-DAG: #define __HAS_LDEXPF__ 1
// GFX700-DAG: #define __HAS_LDEXPF__ 1
// GFX701-DAG: #define __HAS_LDEXPF__ 1
// GFX702-DAG: #define __HAS_LDEXPF__ 1
// GFX703-DAG: #define __HAS_LDEXPF__ 1
// GFX704-DAG: #define __HAS_LDEXPF__ 1
// GFX705-DAG: #define __HAS_LDEXPF__ 1
// GFX801-DAG: #define __HAS_LDEXPF__ 1
// GFX802-DAG: #define __HAS_LDEXPF__ 1
// GFX803-DAG: #define __HAS_LDEXPF__ 1
// GFX805-DAG: #define __HAS_LDEXPF__ 1
// GFX810-DAG: #define __HAS_LDEXPF__ 1
// GFX900-DAG: #define __HAS_LDEXPF__ 1
// GFX902-DAG: #define __HAS_LDEXPF__ 1
// GFX904-DAG: #define __HAS_LDEXPF__ 1
// GFX906-DAG: #define __HAS_LDEXPF__ 1
// GFX908-DAG: #define __HAS_LDEXPF__ 1
// GFX909-DAG: #define __HAS_LDEXPF__ 1
// GFX1010-DAG: #define __HAS_LDEXPF__ 1
// GFX1011-DAG: #define __HAS_LDEXPF__ 1
// GFX1012-DAG: #define __HAS_LDEXPF__ 1
// GFX1030-DAG: #define __HAS_LDEXPF__ 1
// GFX1031-DAG: #define __HAS_LDEXPF__ 1

// GFX600-DAG: #define __gfx600__ 1
// GFX601-DAG: #define __gfx601__ 1
// GFX602-DAG: #define __gfx602__ 1
// GFX700-DAG: #define __gfx700__ 1
// GFX701-DAG: #define __gfx701__ 1
// GFX702-DAG: #define __gfx702__ 1
// GFX703-DAG: #define __gfx703__ 1
// GFX704-DAG: #define __gfx704__ 1
// GFX705-DAG: #define __gfx705__ 1
// GFX801-DAG: #define __gfx801__ 1
// GFX802-DAG: #define __gfx802__ 1
// GFX803-DAG: #define __gfx803__ 1
// GFX805-DAG: #define __gfx805__ 1
// GFX810-DAG: #define __gfx810__ 1
// GFX900-DAG: #define __gfx900__ 1
// GFX902-DAG: #define __gfx902__ 1
// GFX904-DAG: #define __gfx904__ 1
// GFX906-DAG: #define __gfx906__ 1
// GFX908-DAG: #define __gfx908__ 1
// GFX909-DAG: #define __gfx909__ 1
// GFX1010-DAG: #define __gfx1010__ 1
// GFX1011-DAG: #define __gfx1011__ 1
// GFX1012-DAG: #define __gfx1012__ 1
// GFX1030-DAG: #define __gfx1030__ 1
// GFX1031-DAG: #define __gfx1031__ 1

// GFX600-DAG: #define __amdgcn_processor__ "gfx600"
// GFX601-DAG: #define __amdgcn_processor__ "gfx601"
// GFX602-DAG: #define __amdgcn_processor__ "gfx602"
// GFX700-DAG: #define __amdgcn_processor__ "gfx700"
// GFX701-DAG: #define __amdgcn_processor__ "gfx701"
// GFX702-DAG: #define __amdgcn_processor__ "gfx702"
// GFX703-DAG: #define __amdgcn_processor__ "gfx703"
// GFX704-DAG: #define __amdgcn_processor__ "gfx704"
// GFX705-DAG: #define __amdgcn_processor__ "gfx705"
// GFX801-DAG: #define __amdgcn_processor__ "gfx801"
// GFX802-DAG: #define __amdgcn_processor__ "gfx802"
// GFX803-DAG: #define __amdgcn_processor__ "gfx803"
// GFX805-DAG: #define __amdgcn_processor__ "gfx805"
// GFX810-DAG: #define __amdgcn_processor__ "gfx810"
// GFX900-DAG: #define __amdgcn_processor__ "gfx900"
// GFX902-DAG: #define __amdgcn_processor__ "gfx902"
// GFX904-DAG: #define __amdgcn_processor__ "gfx904"
// GFX906-DAG: #define __amdgcn_processor__ "gfx906"
// GFX908-DAG: #define __amdgcn_processor__ "gfx908"
// GFX909-DAG: #define __amdgcn_processor__ "gfx909"
// GFX1010-DAG: #define __amdgcn_processor__ "gfx1010"
// GFX1011-DAG: #define __amdgcn_processor__ "gfx1011"
// GFX1012-DAG: #define __amdgcn_processor__ "gfx1012"
// GFX1030-DAG: #define __amdgcn_processor__ "gfx1030"
// GFX1031-DAG: #define __amdgcn_processor__ "gfx1031"

// GFX600-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX601-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX602-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX700-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX701-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX702-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX703-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX704-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX705-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX801-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX802-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX803-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX805-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX810-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX900-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX902-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX904-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX906-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX908-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX909-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// GFX1010-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
// GFX1011-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
// GFX1012-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
// GFX1030-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
// GFX1031-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32

// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mwavefrontsize64 \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mwavefrontsize64 \
// RUN:   %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mwavefrontsize64 \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mwavefrontsize64 \
// RUN:   -mno-wavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE32 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 -mno-wavefrontsize64 \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx1010 -mno-wavefrontsize64 \
// RUN:   -mwavefrontsize64 %s 2>&1 | FileCheck --check-prefixes=WAVE64 %s
// WAVE64-DAG: #define __AMDGCN_WAVEFRONT_SIZE 64
// WAVE32-DAG: #define __AMDGCN_WAVEFRONT_SIZE 32
