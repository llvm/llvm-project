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
// RUN: %clang -E -dM -target amdgcn -mcpu=hainan %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=oland %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=pitcairn %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=verde %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX601 %s
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
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx801 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX801 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=carrizo %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX801 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=iceland %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=tonga %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX802 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris10 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=polaris11 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX803 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx810 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX810 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=stoney %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX810 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX900 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx902 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX902 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx904 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX904 %s
// RUN: %clang -E -dM -target amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck --check-prefixes=ARCH-GCN,GFX906 %s

// GFX600-DAG: #define FP_FAST_FMA 1
// GFX601-DAG: #define FP_FAST_FMA 1
// GFX700-DAG: #define FP_FAST_FMA 1
// GFX701-DAG: #define FP_FAST_FMA 1
// GFX702-DAG: #define FP_FAST_FMA 1
// GFX703-DAG: #define FP_FAST_FMA 1
// GFX704-DAG: #define FP_FAST_FMA 1
// GFX801-DAG: #define FP_FAST_FMA 1
// GFX802-DAG: #define FP_FAST_FMA 1
// GFX803-DAG: #define FP_FAST_FMA 1
// GFX810-DAG: #define FP_FAST_FMA 1
// GFX900-DAG: #define FP_FAST_FMA 1
// GFX902-DAG: #define FP_FAST_FMA 1
// GFX904-DAG: #define FP_FAST_FMA 1
// GFX906-DAG: #define FP_FAST_FMA 1

// GFX600-DAG: #define FP_FAST_FMAF 1
// GFX601-NOT: #define FP_FAST_FMAF 1
// GFX700-NOT: #define FP_FAST_FMAF 1
// GFX701-DAG: #define FP_FAST_FMAF 1
// GFX702-DAG: #define FP_FAST_FMAF 1
// GFX703-NOT: #define FP_FAST_FMAF 1
// GFX704-NOT: #define FP_FAST_FMAF 1
// GFX801-DAG: #define FP_FAST_FMAF 1
// GFX802-NOT: #define FP_FAST_FMAF 1
// GFX803-NOT: #define FP_FAST_FMAF 1
// GFX810-NOT: #define FP_FAST_FMAF 1
// GFX900-DAG: #define FP_FAST_FMAF 1
// GFX902-DAG: #define FP_FAST_FMAF 1
// GFX904-DAG: #define FP_FAST_FMAF 1
// GFX906-DAG: #define FP_FAST_FMAF 1

// ARCH-GCN-DAG: #define __AMDGCN__ 1
// ARCH-GCN-DAG: #define __AMDGPU__ 1
// ARCH-GCN-DAG: #define __AMD__ 1

// GFX600-DAG: #define __HAS_FMAF__ 1
// GFX601-DAG: #define __HAS_FMAF__ 1
// GFX700-DAG: #define __HAS_FMAF__ 1
// GFX701-DAG: #define __HAS_FMAF__ 1
// GFX702-DAG: #define __HAS_FMAF__ 1
// GFX703-DAG: #define __HAS_FMAF__ 1
// GFX704-DAG: #define __HAS_FMAF__ 1
// GFX801-DAG: #define __HAS_FMAF__ 1
// GFX802-DAG: #define __HAS_FMAF__ 1
// GFX803-DAG: #define __HAS_FMAF__ 1
// GFX810-DAG: #define __HAS_FMAF__ 1
// GFX900-DAG: #define __HAS_FMAF__ 1
// GFX902-DAG: #define __HAS_FMAF__ 1
// GFX904-DAG: #define __HAS_FMAF__ 1
// GFX906-DAG: #define __HAS_FMAF__ 1

// GFX600-DAG: #define __HAS_FP64__ 1
// GFX601-DAG: #define __HAS_FP64__ 1
// GFX700-DAG: #define __HAS_FP64__ 1
// GFX701-DAG: #define __HAS_FP64__ 1
// GFX702-DAG: #define __HAS_FP64__ 1
// GFX703-DAG: #define __HAS_FP64__ 1
// GFX704-DAG: #define __HAS_FP64__ 1
// GFX801-DAG: #define __HAS_FP64__ 1
// GFX802-DAG: #define __HAS_FP64__ 1
// GFX803-DAG: #define __HAS_FP64__ 1
// GFX810-DAG: #define __HAS_FP64__ 1
// GFX900-DAG: #define __HAS_FP64__ 1
// GFX902-DAG: #define __HAS_FP64__ 1
// GFX904-DAG: #define __HAS_FP64__ 1
// GFX906-DAG: #define __HAS_FP64__ 1

// GFX600-DAG: #define __HAS_LDEXPF__ 1
// GFX601-DAG: #define __HAS_LDEXPF__ 1
// GFX700-DAG: #define __HAS_LDEXPF__ 1
// GFX701-DAG: #define __HAS_LDEXPF__ 1
// GFX702-DAG: #define __HAS_LDEXPF__ 1
// GFX703-DAG: #define __HAS_LDEXPF__ 1
// GFX704-DAG: #define __HAS_LDEXPF__ 1
// GFX801-DAG: #define __HAS_LDEXPF__ 1
// GFX802-DAG: #define __HAS_LDEXPF__ 1
// GFX803-DAG: #define __HAS_LDEXPF__ 1
// GFX810-DAG: #define __HAS_LDEXPF__ 1
// GFX900-DAG: #define __HAS_LDEXPF__ 1
// GFX902-DAG: #define __HAS_LDEXPF__ 1
// GFX904-DAG: #define __HAS_LDEXPF__ 1
// GFX906-DAG: #define __HAS_LDEXPF__ 1

// GFX600-DAG: #define __gfx600__ 1
// GFX601-DAG: #define __gfx601__ 1
// GFX700-DAG: #define __gfx700__ 1
// GFX701-DAG: #define __gfx701__ 1
// GFX702-DAG: #define __gfx702__ 1
// GFX703-DAG: #define __gfx703__ 1
// GFX704-DAG: #define __gfx704__ 1
// GFX801-DAG: #define __gfx801__ 1
// GFX802-DAG: #define __gfx802__ 1
// GFX803-DAG: #define __gfx803__ 1
// GFX810-DAG: #define __gfx810__ 1
// GFX900-DAG: #define __gfx900__ 1
// GFX902-DAG: #define __gfx902__ 1
// GFX904-DAG: #define __gfx904__ 1
// GFX906-DAG: #define __gfx906__ 1
