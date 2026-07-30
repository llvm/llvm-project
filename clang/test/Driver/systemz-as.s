// Make sure SystemZ defaults to using the integrated assembler

// RUN: %clang -target s390x-ibm-linux -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s

// RUN: %clang -target s390x-ibm-linux -integrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=IAS %s
// IAS: "-cc1as"{{.*}} "-target-cpu" "z10"

// RUN: %clang -target s390x-ibm-linux -no-integrated-as -### -c %s 2>&1 \
// RUN: | FileCheck -check-prefix=NO-IAS %s
// NO-IAS-NOT: -cc1as
// NO-IAS: "-march=z10"

