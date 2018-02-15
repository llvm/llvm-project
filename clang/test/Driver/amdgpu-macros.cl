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
// RUN: %clang -E -dM -target r600 -mcpu=turks %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,TURKS %s
// RUN: %clang -E -dM -target r600 -mcpu=aruba %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s
// RUN: %clang -E -dM -target r600 -mcpu=cayman %s 2>&1 | FileCheck --check-prefixes=ARCH-R600,CAYMAN %s

// ARCH-R600-DAG: #define __AMD__ 1
// ARCH-R600-DAG: #define __AMDGPU__ 1
// ARCH-R600-DAG: #define __R600__ 1

// R600:    #define __r600__ 1
// R630:    #define __r630__ 1
// RS880:   #define __rs880__ 1
// RV670:   #define __rv670__ 1
// RV710:   #define __rv710__ 1
// RV730:   #define __rv730__ 1
// RV770:   #define __rv770__ 1
// CEDAR:   #define __cedar__ 1
// CYPRESS: #define __cypress__ 1
// JUNIPER: #define __juniper__ 1
// REDWOOD: #define __redwood__ 1
// SUMO:    #define __sumo__ 1
// BARTS:   #define __barts__ 1
// CAICOS:  #define __caicos__ 1
// TURKS:   #define __turks__ 1
// CAYMAN:  #define __cayman__ 1

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

// ARCH-GCN-DAG: #define __AMD__ 1
// ARCH-GCN-DAG: #define __AMDGPU__ 1
// ARCH-GCN-DAG: #define __AMDGCN__ 1

// GFX600: #define __gfx600__ 1
// GFX601: #define __gfx601__ 1
// GFX700: #define __gfx700__ 1
// GFX701: #define __gfx701__ 1
// GFX702: #define __gfx702__ 1
// GFX703: #define __gfx703__ 1
// GFX704: #define __gfx704__ 1
// GFX801: #define __gfx801__ 1
// GFX802: #define __gfx802__ 1
// GFX803: #define __gfx803__ 1
// GFX810: #define __gfx810__ 1
// GFX900: #define __gfx900__ 1
// GFX902: #define __gfx902__ 1
