// Test that both target features fullfp16 and fp16fml are implemented and available correctly

// fullfp16 is off by default for v8a, feature must not be mentioned
// RUN: %clang --target=aarch64 -march=armv8a  -### -c %s 2>&1 | FileCheck -check-prefix=V82ANOFP16 -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=V82ANOFP16 -check-prefix=GENERIC %s
// V82ANOFP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// V82ANOFP16-NOT: "-target-feature" "{{[+-]}}fullfp16"
// GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=aarch64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NO-FP16FML %s
// GENERICV8A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16 %s
// GENERICV8A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-FP16: "-target-feature" "+fullfp16"
// GENERICV8A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-FP16-SAME: {{$}}

// RUN: %clang --target=aarch64 -march=armv8a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16FML %s
// GENERICV8A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML %s
// GENERICV82A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.2a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16 %s
// GENERICV82A-FP16: "-cc1"{{.*}} "-triple" "aarch64"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.2a"{{.*}} "-target-feature" "+fullfp16"{{.*}} "-target-feature" "+neon"
// GENERICV82A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-FP16-SAME: {{$}}

// RUN: %clang --target=aarch64 -march=armv8.2-a+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-SPE %s
// GENERICV82A-SPE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v8.2a"{{.*}} "-target-feature" "+spe"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64 -march=armv8.2a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML %s
// GENERICV82A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.2a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-NO-FP16FML %s
// GENERICV82A-FP16-NO-FP16FML: "-target-feature" "+fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.2a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML-FP16 %s
// GENERICV82A-NO-FP16FML-FP16: "-target-feature" "+fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.2a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML-NO-FP16 %s
// GENERICV82A-FP16FML-NO-FP16: "-target-feature" "-fullfp16"{{.*}} "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.2a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16-FP16FML %s
// GENERICV82A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.2a+fp16+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-SPE %s
// RUN: %clang --target=aarch64 -march=armv8.2-a+fp16+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-SPE %s
// GENERICV82A-FP16-SPE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.2a"{{.*}}  "-target-feature" "+fullfp16"{{.*}} "-target-feature" "+spe"{{.*}} "-target-feature" "+neon"
// GENERICV82A-FP16-SPE-NOT:  "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-FP16-SPE-SAME: {{$}}

// RUN: %clang --target=aarch64 -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML %s
// GENERICV83A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV83A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.3a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16 %s
// GENERICV83A-FP16: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.3a"{{.*}} "-target-feature" "+fullfp16"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64 -march=armv8.3a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML %s
// GENERICV83A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.3a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16-NO-FP16FML %s
// GENERICV83A-FP16-NO-FP16FML: "-target-feature" "+fullfp16"
// GENERICV83A-FP16-NO-FP16FML-NOT: fp16fml

// RUN: %clang --target=aarch64 -march=armv8.3a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML-FP16 %s
// GENERICV83A-NO-FP16FML-FP16: "-target-feature" "+fullfp16"
// GENERICV83A-NO-FP16FML-FP16-NOT: fp16fml

// RUN: %clang --target=aarch64 -march=armv8.3a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML-NO-FP16 %s
// GENERICV83A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.3a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.3-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16-FP16FML %s
// GENERICV83A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML %s
// GENERICV84A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV84A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.4a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16 %s
// GENERICV84A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML %s
// GENERICV84A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16-NO-FP16FML %s
// GENERICV84A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML-FP16 %s
// GENERICV84A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML-NO-FP16 %s
// GENERICV84A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.4a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.4-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16-FP16FML %s
// GENERICV84A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16FML %s
// GENERICV85A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV85A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.5a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16 %s
// GENERICV85A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16FML %s
// GENERICV85A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16-NO-FP16FML %s
// GENERICV85A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16FML-FP16 %s
// GENERICV85A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16FML-NO-FP16 %s
// GENERICV85A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.5a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.5-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-NO-FP16-FP16FML %s
// GENERICV85A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16FML %s
// GENERICV86A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV86A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.6a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16 %s
// GENERICV86A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16FML %s
// GENERICV86A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16-NO-FP16FML %s
// GENERICV86A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16FML-FP16 %s
// GENERICV86A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-FP16FML-NO-FP16 %s
// GENERICV86A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.6a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.6-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-NO-FP16-FP16FML %s
// GENERICV86A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16FML %s
// GENERICV87A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV87A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.7a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16 %s
// GENERICV87A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16FML %s
// GENERICV87A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16-NO-FP16FML %s
// GENERICV87A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16FML-FP16 %s
// GENERICV87A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-FP16FML-NO-FP16 %s
// GENERICV87A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.7a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.7-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-NO-FP16-FP16FML %s
// GENERICV87A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16FML %s
// GENERICV88A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV88A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang --target=aarch64 -march=armv8.8a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16 %s
// GENERICV88A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16FML %s
// GENERICV88A-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16-NO-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16-NO-FP16FML %s
// GENERICV88A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16FML-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16FML-FP16 %s
// GENERICV88A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16FML-NO-FP16 %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-FP16FML-NO-FP16 %s
// GENERICV88A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang --target=aarch64 -march=armv8.8a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16-FP16FML %s
// RUN: %clang --target=aarch64 -march=armv8.8-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-NO-FP16-FP16FML %s
// GENERICV88A-NO-FP16-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"
