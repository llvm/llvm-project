// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=thread | FileCheck -check-prefix=TSAN %s
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s -fsanitize=thread -fsanitize-ignorelist=%t | FileCheck -check-prefix=BL %s

// The sanitize_thread attribute should be attached to functions
// when ThreadSanitizer is enabled, unless no_sanitize_thread attribute
// is present.

// WITHOUT:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// TSAN:  NoTSAN1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize_thread))
int NoTSAN1(int *a) { return *a; }

// WITHOUT:  NoTSAN2{{.*}}) [[NOATTR]]
// BL:  NoTSAN2{{.*}}) [[NOATTR]]
// TSAN:  NoTSAN2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize_thread))
int NoTSAN2(int *a);
int NoTSAN2(int *a) { return *a; }

// WITHOUT:  NoTSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoTSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// TSAN:  NoTSAN3{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize("thread")))
int NoTSAN3(int *a) { return *a; }

// WITHOUT:  TSANOk{{.*}}) [[NOATTR]]
// BL:  TSANOk{{.*}}) [[NOATTR]]
// TSAN: TSANOk{{.*}}) [[WITH:#[0-9]+]]
int TSANOk(int *a) { return *a; }

// WITHOUT:  TemplateTSANOk{{.*}}) [[NOATTR]]
// BL:  TemplateTSANOk{{.*}}) [[NOATTR]]
// TSAN: TemplateTSANOk{{.*}}) [[WITH]]
template<int i>
int TemplateTSANOk() { return i; }

// WITHOUT:  TemplateNoTSAN{{.*}}) [[NOATTR]]
// BL:  TemplateNoTSAN{{.*}}) [[NOATTR]]
// TSAN: TemplateNoTSAN{{.*}}) [[NOATTR]]
template<int i>
__attribute__((no_sanitize_thread))
int TemplateNoTSAN() { return i; }

int force_instance = TemplateTSANOk<42>()
                   + TemplateNoTSAN<42>();

// Check that __cxx_global_var_init* get the sanitize_thread attribute.
int global1 = 0;
int global2 = *(int*)((char*)&global1+1);
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// BL: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// TSAN: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]

// WITHOUT: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// BL: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// TSAN: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }
// TSAN: attributes [[WITH]] = { noinline nounwind sanitize_thread{{.*}} }
