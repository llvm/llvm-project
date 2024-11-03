// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: zlib

// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -g -gz 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf -fdebug-info-for-profiling 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf-2 -gsplit-dwarf 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf-3 -glldb 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf-4 -gcodeview 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf-5 -gmodules 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -ggdb1 -fdebug-macro 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -ggdb2 -ggnu-pubnames 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -ggdb3 -gdwarf-aranges 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -g -gcolumn-info -fdebug-types-section 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN,COMMON

// Same tests for OpenMP
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -g -gz 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf -fdebug-info-for-profiling 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf-2 -gsplit-dwarf 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf-3 -glldb 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf-4 -gcodeview 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf-5 -gmodules 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -ggdb1 -fdebug-macro 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -ggdb2 -ggnu-pubnames 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -ggdb3 -gdwarf-aranges 2>&1 | FileCheck %s --check-prefixes WARN,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -g -gcolumn-info -fdebug-types-section 2>&1 | FileCheck %s --check-prefixes WARN,COMMON

// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -gdwarf-5 -gembed-source 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN-GES,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -c %s -ggdb -gembed-source -gdwarf-5 2>&1 \
// RUN: | FileCheck %s --check-prefixes WARN-GES,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -gdwarf-5 -gembed-source 2>&1 | FileCheck %s --check-prefixes WARN-GES,COMMON
// RUN: not %clang -### --target=x86_64-linux-gnu -fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda -c %s \
// RUN:   -fgpu-rdc -ggdb -gembed-source -gdwarf-5 2>&1 | FileCheck %s --check-prefixes WARN-GES,COMMON

// COMMON: warning: debug information option '{{-gz|-fdebug-info-for-profiling|-gsplit-dwarf|-glldb|-gcodeview|-gmodules|-gembed-source|-fdebug-macro|-ggnu-pubnames|-gdwarf-aranges|-fdebug-types-section}}' is not supported
// WARN-SAME: for target 'nvptx64-nvidia-cuda' [-Wunsupported-target-opt]
// WARN-GES-SAME: requires DWARF-5 but target 'nvptx64-nvidia-cuda' only provides DWARF-2 [-Wunsupported-target-opt]
// COMMON-NOT: debug information option '{{.*}}' is not supported for target 'x86
// COMMON: "-triple" "nvptx64-nvidia-cuda"
// COMMON-NOT: {{-compress-debug|-fdebug-info-for-profiling|lldb|codeview|module-format|embed-source|debug-info-macro|gnu-pubnames|generate-arange-section|generate-type-units}}
// COMMON: "-triple" "x86_64
// COMMON-SAME: {{-compress-debug|-fdebug-info-for-profiling|split-dwarf|lldb|codeview|module-format|embed-source|debug-info-macro|gnu-pubnames|generate-arange-section|generate-type-units}}
