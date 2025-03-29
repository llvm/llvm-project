// AIX does not support -gdwarf-5 which is required by -gembed-source
// UNSUPPORTED: target={{.*}}-aix{{.*}}

// RUN: %clang -### -gdwarf-5 -gembed-source %s 2>&1 | FileCheck -check-prefix=GEMBED_5 %s
// RUN: not %clang -### -gdwarf-2 -gembed-source %s 2>&1 | FileCheck -check-prefix=GEMBED_2 %s
// RUN: %clang -### -gdwarf-5 -gno-embed-source %s 2>&1 | FileCheck -check-prefix=NOGEMBED_5 %s
// RUN: %clang -### -gdwarf-2 -gno-embed-source %s 2>&1 | FileCheck -check-prefix=NOGEMBED_2 %s
//
// GEMBED_5:  "-gembed-source"
// GEMBED_2:  error: invalid argument '-gembed-source' only allowed with '-gdwarf-5'
// NOGEMBED_5-NOT:  "-gembed-source"
// NOGEMBED_2-NOT:  error: invalid argument '-gembed-source' only allowed with '-gdwarf-5'
//
