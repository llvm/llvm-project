// Check that -mcpu works for all supported GPUs.

//
// R600-based processors.
//

// RUN: %clang -### -target r600 -mcpu=r600 %s 2>&1 | FileCheck --check-prefix=R600 %s
// RUN: %clang -### -target r600 -mcpu=rv630 %s 2>&1 | FileCheck --check-prefix=R600 %s
// RUN: %clang -### -target r600 -mcpu=rv635 %s 2>&1 | FileCheck --check-prefix=R600 %s
// RUN: %clang -### -target r600 -mcpu=r630 %s 2>&1 | FileCheck --check-prefix=R630 %s
// RUN: %clang -### -target r600 -mcpu=rs780 %s 2>&1 | FileCheck --check-prefix=RS880 %s
// RUN: %clang -### -target r600 -mcpu=rs880 %s 2>&1 | FileCheck --check-prefix=RS880 %s
// RUN: %clang -### -target r600 -mcpu=rv610 %s 2>&1 | FileCheck --check-prefix=RS880 %s
// RUN: %clang -### -target r600 -mcpu=rv620 %s 2>&1 | FileCheck --check-prefix=RS880 %s
// RUN: %clang -### -target r600 -mcpu=rv670 %s 2>&1 | FileCheck --check-prefix=RV670 %s
// RUN: %clang -### -target r600 -mcpu=rv710 %s 2>&1 | FileCheck --check-prefix=RV710 %s
// RUN: %clang -### -target r600 -mcpu=rv730 %s 2>&1 | FileCheck --check-prefix=RV730 %s
// RUN: %clang -### -target r600 -mcpu=rv740 %s 2>&1 | FileCheck --check-prefix=RV770 %s
// RUN: %clang -### -target r600 -mcpu=rv770 %s 2>&1 | FileCheck --check-prefix=RV770 %s
// RUN: %clang -### -target r600 -mcpu=cedar %s 2>&1 | FileCheck --check-prefix=CEDAR %s
// RUN: %clang -### -target r600 -mcpu=palm %s 2>&1 | FileCheck --check-prefix=CEDAR %s
// RUN: %clang -### -target r600 -mcpu=cypress %s 2>&1 | FileCheck --check-prefix=CYPRESS %s
// RUN: %clang -### -target r600 -mcpu=hemlock %s 2>&1 | FileCheck --check-prefix=CYPRESS %s
// RUN: %clang -### -target r600 -mcpu=juniper %s 2>&1 | FileCheck --check-prefix=JUNIPER %s
// RUN: %clang -### -target r600 -mcpu=redwood %s 2>&1 | FileCheck --check-prefix=REDWOOD %s
// RUN: %clang -### -target r600 -mcpu=sumo %s 2>&1 | FileCheck --check-prefix=SUMO %s
// RUN: %clang -### -target r600 -mcpu=sumo2 %s 2>&1 | FileCheck --check-prefix=SUMO %s
// RUN: %clang -### -target r600 -mcpu=barts %s 2>&1 | FileCheck --check-prefix=BARTS %s
// RUN: %clang -### -target r600 -mcpu=caicos %s 2>&1 | FileCheck --check-prefix=CAICOS %s
// RUN: %clang -### -target r600 -mcpu=aruba %s 2>&1 | FileCheck --check-prefix=CAYMAN %s
// RUN: %clang -### -target r600 -mcpu=cayman %s 2>&1 | FileCheck --check-prefix=CAYMAN %s
// RUN: %clang -### -target r600 -mcpu=turks %s 2>&1 | FileCheck --check-prefix=TURKS %s

// R600:    "-target-cpu" "r600"
// R630:    "-target-cpu" "r630"
// RS880:   "-target-cpu" "rs880"
// RV670:   "-target-cpu" "rv670"
// RV710:   "-target-cpu" "rv710"
// RV730:   "-target-cpu" "rv730"
// RV770:   "-target-cpu" "rv770"
// CEDAR:   "-target-cpu" "cedar"
// CYPRESS: "-target-cpu" "cypress"
// JUNIPER: "-target-cpu" "juniper"
// REDWOOD: "-target-cpu" "redwood"
// SUMO:    "-target-cpu" "sumo"
// BARTS:   "-target-cpu" "barts"
// CAICOS:  "-target-cpu" "caicos"
// CAYMAN:  "-target-cpu" "cayman"
// TURKS:   "-target-cpu" "turks"

//
// AMDGCN-based processors.
//

// RUN: %clang -### -target amdgcn %s 2>&1 | FileCheck --check-prefix=GCNDEFAULT %s
// RUN: %clang -### -target amdgcn -mcpu=gfx600 %s 2>&1 | FileCheck --check-prefix=GFX600 %s
// RUN: %clang -### -target amdgcn -mcpu=tahiti %s 2>&1 | FileCheck --check-prefix=GFX600 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx601 %s 2>&1 | FileCheck --check-prefix=GFX601 %s
// RUN: %clang -### -target amdgcn -mcpu=pitcairn %s 2>&1 | FileCheck --check-prefix=GFX601 %s
// RUN: %clang -### -target amdgcn -mcpu=verde %s 2>&1 | FileCheck --check-prefix=GFX601 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx602 %s 2>&1 | FileCheck --check-prefix=GFX602 %s
// RUN: %clang -### -target amdgcn -mcpu=hainan %s 2>&1 | FileCheck --check-prefix=GFX602 %s
// RUN: %clang -### -target amdgcn -mcpu=oland %s 2>&1 | FileCheck --check-prefix=GFX602 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx700 %s 2>&1 | FileCheck --check-prefix=GFX700 %s
// RUN: %clang -### -target amdgcn -mcpu=kaveri %s 2>&1 | FileCheck --check-prefix=GFX700 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx701 %s 2>&1 | FileCheck --check-prefix=GFX701 %s
// RUN: %clang -### -target amdgcn -mcpu=hawaii %s 2>&1 | FileCheck --check-prefix=GFX701 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx702 %s 2>&1 | FileCheck --check-prefix=GFX702 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx703 %s 2>&1 | FileCheck --check-prefix=GFX703 %s
// RUN: %clang -### -target amdgcn -mcpu=kabini %s 2>&1 | FileCheck --check-prefix=GFX703 %s
// RUN: %clang -### -target amdgcn -mcpu=mullins %s 2>&1 | FileCheck --check-prefix=GFX703 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx704 %s 2>&1 | FileCheck --check-prefix=GFX704 %s
// RUN: %clang -### -target amdgcn -mcpu=bonaire %s 2>&1 | FileCheck --check-prefix=GFX704 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx705 %s 2>&1 | FileCheck --check-prefix=GFX705 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx801 %s 2>&1 | FileCheck --check-prefix=GFX801 %s
// RUN: %clang -### -target amdgcn -mcpu=carrizo %s 2>&1 | FileCheck --check-prefix=GFX801 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx802 %s 2>&1 | FileCheck --check-prefix=GFX802 %s
// RUN: %clang -### -target amdgcn -mcpu=iceland %s 2>&1 | FileCheck --check-prefix=GFX802 %s
// RUN: %clang -### -target amdgcn -mcpu=tonga %s 2>&1 | FileCheck --check-prefix=GFX802 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx803 %s 2>&1 | FileCheck --check-prefix=GFX803 %s
// RUN: %clang -### -target amdgcn -mcpu=fiji %s 2>&1 | FileCheck --check-prefix=GFX803 %s
// RUN: %clang -### -target amdgcn -mcpu=polaris10 %s 2>&1 | FileCheck --check-prefix=GFX803 %s
// RUN: %clang -### -target amdgcn -mcpu=polaris11 %s 2>&1 | FileCheck --check-prefix=GFX803 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx805 %s 2>&1 | FileCheck --check-prefix=GFX805 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx810 %s 2>&1 | FileCheck --check-prefix=GFX810 %s
// RUN: %clang -### -target amdgcn -mcpu=stoney %s 2>&1 | FileCheck --check-prefix=GFX810 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx900 %s 2>&1 | FileCheck --check-prefix=GFX900 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx902 %s 2>&1 | FileCheck --check-prefix=GFX902 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx904 %s 2>&1 | FileCheck --check-prefix=GFX904 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx906 %s 2>&1 | FileCheck --check-prefix=GFX906 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx908 %s 2>&1 | FileCheck --check-prefix=GFX908 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx909 %s 2>&1 | FileCheck --check-prefix=GFX909 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx90a %s 2>&1 | FileCheck --check-prefix=GFX90A %s
// RUN: %clang -### -target amdgcn -mcpu=gfx90c %s 2>&1 | FileCheck --check-prefix=GFX90C %s
// RUN: %clang -### -target amdgcn -mcpu=gfx940 %s 2>&1 | FileCheck --check-prefix=GFX940 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1010 %s 2>&1 | FileCheck --check-prefix=GFX1010 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1011 %s 2>&1 | FileCheck --check-prefix=GFX1011 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1012 %s 2>&1 | FileCheck --check-prefix=GFX1012 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1013 %s 2>&1 | FileCheck --check-prefix=GFX1013 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1030 %s 2>&1 | FileCheck --check-prefix=GFX1030 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1031 %s 2>&1 | FileCheck --check-prefix=GFX1031 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1032 %s 2>&1 | FileCheck --check-prefix=GFX1032 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1033 %s 2>&1 | FileCheck --check-prefix=GFX1033 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1034 %s 2>&1 | FileCheck --check-prefix=GFX1034 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1035 %s 2>&1 | FileCheck --check-prefix=GFX1035 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1036 %s 2>&1 | FileCheck --check-prefix=GFX1036 %s

// GCNDEFAULT-NOT: -target-cpu
// GFX600:    "-target-cpu" "gfx600"
// GFX601:    "-target-cpu" "gfx601"
// GFX602:    "-target-cpu" "gfx602"
// GFX700:    "-target-cpu" "gfx700"
// GFX701:    "-target-cpu" "gfx701"
// GFX702:    "-target-cpu" "gfx702"
// GFX703:    "-target-cpu" "gfx703"
// GFX704:    "-target-cpu" "gfx704"
// GFX705:    "-target-cpu" "gfx705"
// GFX801:    "-target-cpu" "gfx801"
// GFX802:    "-target-cpu" "gfx802"
// GFX803:    "-target-cpu" "gfx803"
// GFX805:    "-target-cpu" "gfx805"
// GFX810:    "-target-cpu" "gfx810"
// GFX900:    "-target-cpu" "gfx900"
// GFX902:    "-target-cpu" "gfx902"
// GFX904:    "-target-cpu" "gfx904"
// GFX906:    "-target-cpu" "gfx906"
// GFX908:    "-target-cpu" "gfx908"
// GFX909:    "-target-cpu" "gfx909"
// GFX90A:    "-target-cpu" "gfx90a"
// GFX90C:    "-target-cpu" "gfx90c"
// GFX940:    "-target-cpu" "gfx940"
// GFX1010:   "-target-cpu" "gfx1010"
// GFX1011:   "-target-cpu" "gfx1011"
// GFX1012:   "-target-cpu" "gfx1012"
// GFX1013:   "-target-cpu" "gfx1013"
// GFX1030:   "-target-cpu" "gfx1030"
// GFX1031:   "-target-cpu" "gfx1031"
// GFX1032:   "-target-cpu" "gfx1032"
// GFX1033:   "-target-cpu" "gfx1033"
// GFX1034:   "-target-cpu" "gfx1034"
// GFX1035:   "-target-cpu" "gfx1035"
// GFX1036:   "-target-cpu" "gfx1036"
