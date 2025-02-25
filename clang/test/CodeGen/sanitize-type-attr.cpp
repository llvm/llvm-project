// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck -check-prefix=WITHOUT %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -fsanitize=type | FileCheck -check-prefix=TYSAN %s
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s -fsanitize=type -fsanitize-blacklist=%t | FileCheck -check-prefix=BL %s

// The sanitize_type attribute should be attached to functions
// when TypeSanitizer is enabled, unless no_sanitize("type") attribute
// is present.

// WITHOUT:  NoTYSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoTYSAN1{{.*}}) [[NOATTR:#[0-9]+]]
// TYSAN:  NoTYSAN1{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize("type"))) int NoTYSAN1(int *a) { return *a; }

// WITHOUT:  NoTYSAN2{{.*}}) [[NOATTR]]
// BL:  NoTYSAN2{{.*}}) [[NOATTR]]
// TYSAN:  NoTYSAN2{{.*}}) [[NOATTR]]
__attribute__((no_sanitize("type"))) int NoTYSAN2(int *a);
int NoTYSAN2(int *a) { return *a; }

// WITHOUT:  NoTYSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// BL:  NoTYSAN3{{.*}}) [[NOATTR:#[0-9]+]]
// TYSAN:  NoTYSAN3{{.*}}) [[NOATTR:#[0-9]+]]
__attribute__((no_sanitize("type"))) int NoTYSAN3(int *a) { return *a; }

// WITHOUT:  TYSANOk{{.*}}) [[NOATTR]]
// BL:  TYSANOk{{.*}}) [[NOATTR]]
// TYSAN: TYSANOk{{.*}}) [[WITH:#[0-9]+]]
int TYSANOk(int *a) { return *a; }

// WITHOUT:  TemplateTYSANOk{{.*}}) [[NOATTR]]
// BL:  TemplateTYSANOk{{.*}}) [[NOATTR]]
// TYSAN: TemplateTYSANOk{{.*}}) [[WITH]]
template <int i>
int TemplateTYSANOk() { return i; }

// WITHOUT:  TemplateNoTYSAN{{.*}}) [[NOATTR]]
// BL:  TemplateNoTYSAN{{.*}}) [[NOATTR]]
// TYSAN: TemplateNoTYSAN{{.*}}) [[NOATTR]]
template <int i>
__attribute__((no_sanitize("type"))) int TemplateNoTYSAN() { return i; }

int force_instance = TemplateTYSANOk<42>() + TemplateNoTYSAN<42>();

// Check that __cxx_global_var_init* get the sanitize_type attribute.
int global1 = 0;
int global2 = *(int *)((char *)&global1 + 1);
// WITHOUT: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// BL: @__cxx_global_var_init{{.*}}[[NOATTR:#[0-9]+]]
// TYSAN: @__cxx_global_var_init{{.*}}[[WITH:#[0-9]+]]

// Make sure that we don't add globals to the list for which we don't have a
// specific type description.
// FIXME: We now have a type description for this type and a global is added. Should it?
struct SX {
  int a, b;
};
SX sx;

void consumer(const char *);

void char_caller() {
  // TYSAN: void @_Z11char_callerv()
  // TYSAN-NEXT: entry:
  // TYSAN-NEXT: call void @_Z8consumerPKc(ptr noundef @.str)
  // TYSAN-NEXT: ret void

  consumer("foo");
}

// WITHOUT: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// BL: attributes [[NOATTR]] = { noinline nounwind{{.*}} }

// TYSAN: attributes [[NOATTR]] = { mustprogress noinline nounwind{{.*}} }
// TYSAN: attributes [[WITH]] = { noinline nounwind sanitize_type{{.*}} }

// TYSAN-DAG: !llvm.tysan.globals = !{[[G1MD:![0-9]+]], [[G2MD:![0-9]+]], [[G3MD:![0-9]+]], [[SXMD:![0-9]+]]}
// TYSAN-DAG: [[G1MD]] = !{ptr @force_instance, [[INTMD:![0-9]+]]}
// TYSAN-DAG: [[INTMD]] = !{!"int",
// TYSAN-DAG: [[G2MD]] = !{ptr @global1, [[INTMD]]}
// TYSAN-DAG: [[G3MD]] = !{ptr @global2, [[INTMD]]}
// TYSAN-DAG: [[SXMD]] = !{ptr @sx, [[SXTYMD:![0-9]+]]}
// TYSAN-DAG: [[SXTYMD]] = !{!"_ZTS2SX", [[INTMD]], i64 0, !1, i64 4}
// TYSAN-DAG: Simple C++ TBAA
