// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=ATTR
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme \
// RUN:   -DSTREAMING -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=STREAM

#ifndef STREAMING
// ATTR-LABEL: @default_before()
// ATTR-SAME: #[[DEFAULT:[0-9]+]]
void default_before(void) {}

// ATTR-LABEL: @with_sve()
// ATTR-SAME: #[[SVE:[0-9]+]]
__attribute__((target("+sve"))) void with_sve(void) {}

// ATTR-LABEL: @default_after()
// ATTR-SAME: #[[DEFAULT]]
void default_after(void) {}

// ATTR-LABEL: @without_sve()
// ATTR-SAME: #[[DEFAULT]]
__attribute__((target("nosve"))) void without_sve(void) {}

// ATTR-LABEL: @default_again()
// ATTR-SAME: #[[DEFAULT]]
void default_again(void) {}

// ATTR: attributes #[[DEFAULT]] = {
// ATTR-NOT: vscale_range
// ATTR: attributes #[[SVE]] = { {{.*}}vscale_range(1,16)
#else
// STREAM-LABEL: @normal_before()
// STREAM-SAME: #[[NORMAL:[0-9]+]]
void normal_before(void) {}

// STREAM-LABEL: @streaming()
// STREAM-SAME: #[[STREAMING:[0-9]+]]
void streaming(void) __arm_streaming {}

// STREAM-LABEL: @compatible()
// STREAM-SAME: #[[COMPATIBLE:[0-9]+]]
void compatible(void) __arm_streaming_compatible {}

// STREAM-LABEL: @locally_streaming()
// STREAM-SAME: #[[LOCAL:[0-9]+]]
__arm_locally_streaming void locally_streaming(void) {}

// STREAM-LABEL: @normal_after()
// STREAM-SAME: #[[NORMAL]]
void normal_after(void) {}

// STREAM: attributes #[[NORMAL]] = {
// STREAM-NOT: vscale_range
// STREAM: attributes #[[STREAMING]] = { {{.*}}vscale_range(1,16)
// STREAM: attributes #[[COMPATIBLE]] = {
// STREAM-NOT: vscale_range
// STREAM: attributes #[[LOCAL]] = { {{.*}}vscale_range(1,16)
#endif
