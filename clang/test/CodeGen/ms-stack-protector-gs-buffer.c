// Check the attributes clang-cl's default /GS emits: sspstrong paired with the
// "stack-protector-gs-buffer" marker that selects MSVC's GS-buffer heuristic.
//
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -O2 \
// RUN:     -stack-protector 4 -emit-llvm %s -o - | FileCheck %s --check-prefix=GS
//
// MSVC does not insert buffer security checks when optimizations are disabled.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions \
// RUN:     -stack-protector 4 -emit-llvm %s -o - | FileCheck %s --check-prefix=NOOPT
//
// An explicit GCC-style level is unaffected by any of this.
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -O2 \
// RUN:     -stack-protector 2 -emit-llvm %s -o - | FileCheck %s --check-prefix=STRONG

void use(void *);

// GS:     define dso_local void @plain() {{.*}}#[[#PLAIN:]] {
// NOOPT:  define dso_local void @plain() #[[#PLAIN:]] {
// STRONG: define dso_local void @plain() {{.*}}#[[#SPLAIN:]] {
void plain(void) { char buf[64]; use(buf); }

// __declspec(safebuffers) opts out entirely, at every level.
// GS:     define dso_local void @safe() {{.*}}#[[#SAFE:]] {
// STRONG: define dso_local void @safe() {{.*}}#[[#SSAFE:]] {
__declspec(safebuffers) void safe(void) { char buf[64]; use(buf); }

// __declspec(strict_gs_check) asks for a cookie in a greater number of
// functions, so it opts back out of the narrower GS-buffer rules and is not
// suppressed at -O0.
// GS:     define dso_local void @strict() {{.*}}#[[#STRICT:]] {
// NOOPT:  define dso_local void @strict() #[[#STRICT:]] {
__declspec(strict_gs_check) void strict(void) { char buf[64]; use(buf); }

// The string attribute sorts between "stack-protector-buffer-size" and
// "target-features", so matching those neighbours proves it is absent.
//
// GS:      attributes #[[#PLAIN]] = { nounwind sspstrong {{.*}}"stack-protector-buffer-size"="8" "stack-protector-gs-buffer"="true" "target-features"
// GS:      attributes #[[#SAFE]] = { nounwind "min-legal-vector-width"
// GS:      attributes #[[#STRICT]] = { nounwind sspstrong {{.*}}"stack-protector-buffer-size"="8" "target-features"

// At -O0 the GS-buffer default produces no stack protector attribute at all,
// so @plain shares an attribute group with @safe.
// NOOPT:   attributes #[[#PLAIN]] = { noinline nounwind optnone "min-legal-vector-width"
// NOOPT:   attributes #[[#STRICT]] = { noinline nounwind optnone sspstrong {{.*}}"stack-protector-buffer-size"="8" "target-features"

// -stack-protector 2 keeps the GCC-compatible strong heuristic: no marker.
// STRONG:  attributes #[[#SPLAIN]] = { nounwind sspstrong {{.*}}"stack-protector-buffer-size"="8" "target-features"
// STRONG:  attributes #[[#SSAFE]] = { nounwind "min-legal-vector-width"
