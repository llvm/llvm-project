// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=numerical | FileCheck -check-prefix=NSAN %s
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=numerical -fsanitize-ignorelist=%t | FileCheck -check-prefix=BL %s

// WITHOUT:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// NSAN:  NoNSAN3{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize("numerical")))
int NoNSAN3(int *a) { return *a; }

// WITHOUT:  NSANOk{{.*}}) [[NOATTR]]
// BL:  NSANOk{{.*}}) [[NOATTR]]
// NSAN: NSANOk{{.*}}) [[WITH:#[0-9]+]]
int NSANOk(int *a) { return *a; }

// WITHOUT:  TemplateNSANOk{{.*}}) [[NOATTR]]
// BL:  TemplateNSANOk{{.*}}) [[NOATTR]]
// NSAN: TemplateNSANOk{{.*}}) [[WITH]]
template<int i>
int TemplateNSANOk() { return i; }

// WITHOUT:  TemplateNoNSAN{{.*}}) [[NOATTR]]
// BL:  TemplateNoNSAN{{.*}}) [[NOATTR]]
// NSAN: TemplateNoNSAN{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize("numerical")))
int TemplateNoNSAN() { return i; }

int force_instance = TemplateNSANOk<42>() + TemplateNoNSAN<42>();

// WITHOUT: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }
// BL: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }
// NSAN: attributes [[WITH]] = { mustprogress noinline nounwind optnone sanitize_numerical_stability{{.*}} }
