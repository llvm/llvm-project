// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: zlib

// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -g -gz 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -gdwarf -fdebug-info-for-profiling 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -gdwarf-2 -gsplit-dwarf 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -gdwarf-3 -glldb 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -gdwarf-4 -gcodeview 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -gdwarf-5 -gmodules 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -ggdb -gembed-source -gdwarf-5 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -ggdb1 -fdebug-macro 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -ggdb2 -ggnu-pubnames 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -ggdb3 -gdwarf-aranges 2>&1 | FileCheck %s
// RUN: %clang -### -target x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s -g -gcolumn-info -fdebug-types-section 2>&1 | FileCheck %s
// CHECK: debug information option '{{-gz|-fdebug-info-for-profiling|-gsplit-dwarf|-glldb|-gcodeview|-gmodules|-gembed-source|-fdebug-macro|-ggnu-pubnames|-gdwarf-aranges|-fdebug-types-section}}' is not supported for target 'nvptx64-nvidia-cuda' [-Wunsupported-target-opt]
// CHECK-NOT: debug information option '{{.*}}' is not supported for target 'x86
// CHECK: "-triple" "nvptx64-nvidia-cuda"
// CHECK-NOT: {{-compress-debug|-fdebug-info-for-profiling|split-dwarf|lldb|codeview|module-format|embed-source|debug-info-macro|gnu-pubnames|generate-arange-section|generate-type-units}}
// CHECK: "-triple" "x86_64
// CHECK-SAME: {{-compress-debug|-fdebug-info-for-profiling|split-dwarf|lldb|codeview|module-format|embed-source|debug-info-macro|gnu-pubnames|generate-arange-section|generate-type-units}}
