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
// RUN: %clang -### -target amdgcn -mcpu=gfx942 %s 2>&1 | FileCheck --check-prefix=GFX942 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx950 %s 2>&1 | FileCheck --check-prefix=GFX950 %s
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
// RUN: %clang -### -target amdgcn -mcpu=gfx1100 %s 2>&1 | FileCheck --check-prefix=GFX1100 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1101 %s 2>&1 | FileCheck --check-prefix=GFX1101 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1102 %s 2>&1 | FileCheck --check-prefix=GFX1102 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1103 %s 2>&1 | FileCheck --check-prefix=GFX1103 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1150 %s 2>&1 | FileCheck --check-prefix=GFX1150 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1151 %s 2>&1 | FileCheck --check-prefix=GFX1151 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1152 %s 2>&1 | FileCheck --check-prefix=GFX1152 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1153 %s 2>&1 | FileCheck --check-prefix=GFX1153 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1154 %s 2>&1 | FileCheck --check-prefix=GFX1154 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1170 %s 2>&1 | FileCheck --check-prefix=GFX1170 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1171 %s 2>&1 | FileCheck --check-prefix=GFX1171 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1172 %s 2>&1 | FileCheck --check-prefix=GFX1172 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1200 %s 2>&1 | FileCheck --check-prefix=GFX1200 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1201 %s 2>&1 | FileCheck --check-prefix=GFX1201 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1250 %s 2>&1 | FileCheck --check-prefix=GFX1250 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1251 %s 2>&1 | FileCheck --check-prefix=GFX1251 %s
// RUN: %clang -### -target amdgcn -mcpu=gfx1310 %s 2>&1 | FileCheck --check-prefix=GFX1310 %s

// RUN: %clang -### -target amdgcn -mcpu=gfx9-generic %s 2>&1 | FileCheck --check-prefix=GFX9_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx9-4-generic %s 2>&1 | FileCheck --check-prefix=GFX9_4_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx10-1-generic %s 2>&1 | FileCheck --check-prefix=GFX10_1_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx10-3-generic %s 2>&1 | FileCheck --check-prefix=GFX10_3_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx11-generic %s 2>&1 | FileCheck --check-prefix=GFX11_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx11-7-generic %s 2>&1 | FileCheck --check-prefix=GFX11_7_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx12-generic %s 2>&1 | FileCheck --check-prefix=GFX12_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx12-5-generic %s 2>&1 | FileCheck --check-prefix=GFX12_5_GENERIC %s
// RUN: %clang -### -target amdgcn -mcpu=gfx13-generic %s 2>&1 | FileCheck --check-prefix=GFX13_GENERIC %s

// GCNDEFAULT: "-triple" "amdgpu--"
// GFX600:    "-triple" "amdgpu6.00--"
// GFX601:    "-triple" "amdgpu6.01--"
// GFX602:    "-triple" "amdgpu6.02--"
// GFX700:    "-triple" "amdgpu7.00--"
// GFX701:    "-triple" "amdgpu7.01--"
// GFX702:    "-triple" "amdgpu7.02--"
// GFX703:    "-triple" "amdgpu7.03--"
// GFX704:    "-triple" "amdgpu7.04--"
// GFX705:    "-triple" "amdgpu7.05--"
// GFX801:    "-triple" "amdgpu8.01--"
// GFX802:    "-triple" "amdgpu8.02--"
// GFX803:    "-triple" "amdgpu8.03--"
// GFX805:    "-triple" "amdgpu8.05--"
// GFX810:    "-triple" "amdgpu8.10--"
// GFX900:    "-triple" "amdgpu9.00--"
// GFX902:    "-triple" "amdgpu9.02--"
// GFX904:    "-triple" "amdgpu9.04--"
// GFX906:    "-triple" "amdgpu9.06--"
// GFX908:    "-triple" "amdgpu9.08--"
// GFX909:    "-triple" "amdgpu9.09--"
// GFX90A:    "-triple" "amdgpu9.0a--"
// GFX90C:    "-triple" "amdgpu9.0c--"
// GFX942:    "-triple" "amdgpu9.42--"
// GFX950:    "-triple" "amdgpu9.50--"
// GFX1010:   "-triple" "amdgpu10.10--"
// GFX1011:   "-triple" "amdgpu10.11--"
// GFX1012:   "-triple" "amdgpu10.12--"
// GFX1013:   "-triple" "amdgpu10.13--"
// GFX1030:   "-triple" "amdgpu10.30--"
// GFX1031:   "-triple" "amdgpu10.31--"
// GFX1032:   "-triple" "amdgpu10.32--"
// GFX1033:   "-triple" "amdgpu10.33--"
// GFX1034:   "-triple" "amdgpu10.34--"
// GFX1035:   "-triple" "amdgpu10.35--"
// GFX1036:   "-triple" "amdgpu10.36--"
// GFX1100:   "-triple" "amdgpu11.00--"
// GFX1101:   "-triple" "amdgpu11.01--"
// GFX1102:   "-triple" "amdgpu11.02--"
// GFX1103:   "-triple" "amdgpu11.03--"
// GFX1150:   "-triple" "amdgpu11.50--"
// GFX1151:   "-triple" "amdgpu11.51--"
// GFX1152:   "-triple" "amdgpu11.52--"
// GFX1153:   "-triple" "amdgpu11.53--"
// GFX1154:   "-triple" "amdgpu11.54--"
// GFX1170:   "-triple" "amdgpu11.70--"
// GFX1171:   "-triple" "amdgpu11.71--"
// GFX1172:   "-triple" "amdgpu11.72--"
// GFX1200:   "-triple" "amdgpu12.00--"
// GFX1201:   "-triple" "amdgpu12.01--"
// GFX1250:   "-triple" "amdgpu12.50--"
// GFX1251:   "-triple" "amdgpu12.51--"
// GFX1310:   "-triple" "amdgpu13.10--"

// GFX9_GENERIC:      "-triple" "amdgpu9--"
// GFX9_4_GENERIC:    "-triple" "amdgpu9.4--"
// GFX10_1_GENERIC:   "-triple" "amdgpu10.1--"
// GFX10_3_GENERIC:   "-triple" "amdgpu10.3--"
// GFX11_GENERIC:     "-triple" "amdgpu11--"
// GFX11_7_GENERIC:   "-triple" "amdgpu11.7--"
// GFX12_GENERIC:     "-triple" "amdgpu12--"
// GFX12_5_GENERIC:   "-triple" "amdgpu12.5--"
// GFX13_GENERIC:     "-triple" "amdgpu13--"
