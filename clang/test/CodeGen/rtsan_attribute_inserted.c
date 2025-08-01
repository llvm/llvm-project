// RUN: %clang_cc1  -triple x86_64-unknown-linux -fsanitize=realtime %s -emit-llvm -o - %s | FileCheck %s

float process(float *a) [[clang::nonblocking]] { return *a; }
// CHECK: @process{{.*}} #0 {
// CHECK: attributes #0 = {
// CHECK-SAME: {{.*sanitize_realtime .*}}

int spinlock(int *a) [[clang::blocking]] { return *a; }
// CHECK: @spinlock{{.*}} #1 {
// CHECK: attributes #1 = {
// CHECK-SAME: {{.*sanitize_realtime_blocking .*}}
