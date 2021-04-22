// RUN: %clang -target powerpc-ibm-aix-xcoff -### -flto=thin 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECKTHINLTO %s
// RUN: %clang -target powerpc64-ibm-aix-xcoff -### -flto=thin 2>&1 %s | \
// RUN:   FileCheck --check-prefix=CHECKTHINLTO %s

// CHECKTHINLTO: error: the clang compiler does not support 'thinLTO on AIX'

