t// Check that -mcpu works for all supported GPUs

// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=r600 %s -o - 2>&1 | FileCheck --check-prefix=R600-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv630 %s -o - 2>&1 | FileCheck --check-prefix=R600-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv635 %s -o - 2>&1 | FileCheck --check-prefix=R600-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv610 %s -o - 2>&1 | FileCheck --check-prefix=RS880-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv620 %s -o - 2>&1 | FileCheck --check-prefix=RS880-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rs780 %s -o - 2>&1 | FileCheck --check-prefix=RS880-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rs880 %s -o - 2>&1 | FileCheck --check-prefix=RS880-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv670 %s -o - 2>&1 | FileCheck --check-prefix=RV670-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv710 %s -o - 2>&1 | FileCheck --check-prefix=RV710-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv730 %s -o - 2>&1 | FileCheck --check-prefix=RV730-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv740 %s -o - 2>&1 | FileCheck --check-prefix=RV770-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=rv770 %s -o - 2>&1 | FileCheck --check-prefix=RV770-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=palm %s -o - 2>&1 | FileCheck --check-prefix=CEDAR-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=cedar %s -o - 2>&1 | FileCheck --check-prefix=CEDAR-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=sumo %s -o - 2>&1 | FileCheck --check-prefix=SUMO-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=sumo2 %s -o - 2>&1 | FileCheck --check-prefix=SUMO-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=redwood %s -o - 2>&1 | FileCheck --check-prefix=REDWOOD-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=juniper %s -o - 2>&1 | FileCheck --check-prefix=JUNIPER-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=juniper %s -o - 2>&1 | FileCheck --check-prefix=JUNIPER-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=hemlock %s -o - 2>&1 | FileCheck --check-prefix=CYPRESS-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=cypress %s -o - 2>&1 | FileCheck --check-prefix=CYPRESS-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=barts %s -o - 2>&1 | FileCheck --check-prefix=BARTS-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=turks %s -o - 2>&1 | FileCheck --check-prefix=TURKS-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=caicos %s -o - 2>&1 | FileCheck --check-prefix=CAICOS-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=cayman %s -o - 2>&1 | FileCheck --check-prefix=CAYMAN-CHECK %s
// RUN: %clang -### -target r600 -x cl -S -emit-llvm -mcpu=aruba %s -o - 2>&1 | FileCheck --check-prefix=CAYMAN-CHECK %s

// R600-CHECK:  "-target-cpu" "r600"
// RS880-CHECK: "-target-cpu" "rs880"
// RV670-CHECK: "-target-cpu" "rv670"
// RV710-CHECK: "-target-cpu" "rv710"
// RV730-CHECK: "-target-cpu" "rv730"
// RV770-CHECK: "-target-cpu" "rv770"
// CEDAR-CHECK: "-target-cpu" "cedar"
// REDWOOD-CHECK: "-target-cpu" "redwood"
// SUMO-CHECK: "-target-cpu" "sumo"
// JUNIPER-CHECK: "-target-cpu" "juniper"
// CYPRESS-CHECK: "-target-cpu" "cypress"
// BARTS-CHECK: "-target-cpu" "barts"
// TURKS-CHECK: "-target-cpu" "turks"
// CAICOS-CHECK: "-target-cpu" "caicos"
// CAYMAN-CHECK: "-target-cpu" "cayman"

// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx600 %s -o - 2>&1 | FileCheck --check-prefix=GFX600-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=tahiti %s -o - 2>&1 | FileCheck --check-prefix=TAHITI-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx601 %s -o - 2>&1 | FileCheck --check-prefix=GFX601-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=pitcairn %s -o - 2>&1 | FileCheck --check-prefix=PITCAIRN-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=verde %s -o - 2>&1 | FileCheck --check-prefix=VERDE-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=oland %s -o - 2>&1 | FileCheck --check-prefix=OLAND-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=hainan %s -o - 2>&1 | FileCheck --check-prefix=HAINAN-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx700 %s -o - 2>&1 | FileCheck --check-prefix=GFX700-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=bonaire %s -o - 2>&1 | FileCheck --check-prefix=BONAIRE-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kaveri %s -o - 2>&1 | FileCheck --check-prefix=KAVERI-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx701 %s -o - 2>&1 | FileCheck --check-prefix=GFX701-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=hawaii %s -o - 2>&1 | FileCheck --check-prefix=HAWAII-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx702 %s -o - 2>&1 | FileCheck --check-prefix=GFX702-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx703 %s -o - 2>&1 | FileCheck --check-prefix=GFX703-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=kabini %s -o - 2>&1 | FileCheck --check-prefix=KABINI-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=mullins %s -o - 2>&1 | FileCheck --check-prefix=MULLINS-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx800 %s -o - 2>&1 | FileCheck --check-prefix=GFX800-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=iceland %s -o - 2>&1 | FileCheck --check-prefix=ICELAND-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx801 %s -o - 2>&1 | FileCheck --check-prefix=GFX801-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=carrizo %s -o - 2>&1 | FileCheck --check-prefix=CARRIZO-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx802 %s -o - 2>&1 | FileCheck --check-prefix=GFX802-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=tonga %s -o - 2>&1 | FileCheck --check-prefix=TONGA-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx803 %s -o - 2>&1 | FileCheck --check-prefix=GFX803-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=fiji %s -o - 2>&1 | FileCheck --check-prefix=FIJI-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=polaris10 %s -o - 2>&1 | FileCheck --check-prefix=POLARIS10-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=polaris11 %s -o - 2>&1 | FileCheck --check-prefix=POLARIS11-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx804 %s -o - 2>&1 | FileCheck --check-prefix=GFX804-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx810 %s -o - 2>&1 | FileCheck --check-prefix=GFX810-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=stoney %s -o - 2>&1 | FileCheck --check-prefix=STONEY-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx900 %s -o - 2>&1 | FileCheck --check-prefix=GFX900-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx901 %s -o - 2>&1 | FileCheck --check-prefix=GFX901-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx902 %s -o - 2>&1 | FileCheck --check-prefix=GFX902-CHECK %s
// RUN: %clang -### -target amdgcn -x cl -S -emit-llvm -mcpu=gfx903 %s -o - 2>&1 | FileCheck --check-prefix=GFX903-CHECK %s

// GFX600-CHECK: "-target-cpu" "gfx600"
// TAHITI-CHECK: "-target-cpu" "tahiti"
// GFX601-CHECK: "-target-cpu" "gfx601"
// PITCAIRN-CHECK: "-target-cpu" "pitcairn"
// VERDE-CHECK: "-target-cpu" "verde"
// OLAND-CHECK: "-target-cpu" "oland"
// HAINAN-CHECK: "-target-cpu" "hainan"
// GFX700-CHECK: "-target-cpu" "gfx700"
// BONAIRE-CHECK: "-target-cpu" "bonaire"
// KAVERI-CHECK: "-target-cpu" "kaveri"
// GFX701-CHECK: "-target-cpu" "gfx701"
// HAWAII-CHECK: "-target-cpu" "hawaii"
// GFX702-CHECK: "-target-cpu" "gfx702"
// GFX703-CHECK: "-target-cpu" "gfx703"
// KABINI-CHECK: "-target-cpu" "kabini"
// MULLINS-CHECK: "-target-cpu" "mullins"
// GFX800-CHECK: "-target-cpu" "gfx800"
// ICELAND-CHECK: "-target-cpu" "iceland"
// GFX801-CHECK: "-target-cpu" "gfx801"
// CARRIZO-CHECK: "-target-cpu" "carrizo"
// GFX802-CHECK: "-target-cpu" "gfx802"
// TONGA-CHECK: "-target-cpu" "tonga"
// GFX803-CHECK: "-target-cpu" "gfx803"
// FIJI-CHECK: "-target-cpu" "fiji"
// POLARIS10-CHECK: "-target-cpu" "polaris10"
// POLARIS11-CHECK: "-target-cpu" "polaris11"
// GFX804-CHECK: "-target-cpu" "gfx804"
// GFX810-CHECK: "-target-cpu" "gfx810"
// STONEY-CHECK: "-target-cpu" "stoney"
// GFX900-CHECK: "-target-cpu" "gfx900"
// GFX901-CHECK: "-target-cpu" "gfx901"
// GFX902-CHECK: "-target-cpu" "gfx902"
// GFX903-CHECK: "-target-cpu" "gfx903"
