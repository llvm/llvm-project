// Tests without serialization:
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -ast-dump %s \
// RUN:   | FileCheck --match-full-lines %s
//
// Tests with serialization:
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-device \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck --match-full-lines %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++17 -triple x86_64-unknown-unknown -fsycl-is-host \
// RUN:   -include-pch %t -ast-dump-all /dev/null \
// RUN:   | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN:   | FileCheck --match-full-lines %s

// These tests validate the AST produced for functions declared with the
// sycl_kernel_entry_point attribute.

// CHECK: TranslationUnitDecl {{.*}}

// A unique kernel name type is required for each declared kernel entry point.
template<int, int=0> struct KN;

__attribute__((sycl_kernel_entry_point(KN<1>)))
void skep1() {
}
// CHECK:      |-FunctionDecl {{.*}} skep1 'void ()'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KN<1>

using KN2 = KN<2>;
__attribute__((sycl_kernel_entry_point(KN2)))
void skep2() {
}
// CHECK:      |-FunctionDecl {{.*}} skep2 'void ()'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KN2

template<int I> using KNT = KN<I>;
__attribute__((sycl_kernel_entry_point(KNT<3>)))
void skep3() {
}
// CHECK:      |-FunctionDecl {{.*}} skep3 'void ()'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KNT<3>

template<typename KNT, typename F>
[[clang::sycl_kernel_entry_point(KNT)]]
void skep4(F f) {
  f();
}
// CHECK:      |-FunctionTemplateDecl {{.*}} skep4
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KNT
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} F
// CHECK-NEXT: | |-FunctionDecl {{.*}} skep4 'void (F)'
// CHECK:      | | `-SYCLKernelEntryPointAttr {{.*}} KNT

void test_skep4() {
  skep4<KNT<4>>([]{});
}
// CHECK:      | `-FunctionDecl {{.*}} used skep4 'void ((lambda at {{.*}}))' implicit_instantiation
// CHECK-NEXT: |   |-TemplateArgument type 'KN<4>'
// CHECK:      |   |-TemplateArgument type '(lambda at {{.*}})'
// CHECK:      |   `-SYCLKernelEntryPointAttr {{.*}} struct KN<4>
// CHECK-NEXT: |-FunctionDecl {{.*}} test_skep4 'void ()'

template<typename KNT, typename T>
[[clang::sycl_kernel_entry_point(KNT)]]
void skep5(T) {
}
// CHECK:      |-FunctionTemplateDecl {{.*}} skep5
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} KNT
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} T
// CHECK-NEXT: | |-FunctionDecl {{.*}} skep5 'void (T)'
// CHECK:      | | `-SYCLKernelEntryPointAttr {{.*}} KNT

// Checks for the explicit template instantiation declaration below.
// CHECK:      | `-FunctionDecl {{.*}} skep5 'void (int)' explicit_instantiation_definition
// CHECK-NEXT: |   |-TemplateArgument type 'KN<5, 4>'
// CHECK:      |   |-TemplateArgument type 'int'
// CHECK:      |   `-SYCLKernelEntryPointAttr {{.*}} KN<5, 4>

// FIXME: C++23 [temp.expl.spec]p12 states:
// FIXME:   ... Similarly, attributes appearing in the declaration of a template
// FIXME:   have no effect on an explicit specialization of that template.
// FIXME: Clang currently instantiates and propagates attributes from a function
// FIXME: template to its explicit specializations resulting in the following
// FIXME: explicit specialization having an attribute incorrectly attached.
template<>
void skep5<KN<5,1>>(short) {
}
// CHECK:      |-FunctionDecl {{.*}} prev {{.*}} skep5 'void (short)' explicit_specialization
// CHECK-NEXT: | |-TemplateArgument type 'KN<5, 1>'
// CHECK:      | |-TemplateArgument type 'short'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} Inherited struct KN<5, 1>

template<>
[[clang::sycl_kernel_entry_point(KN<5,2>)]]
void skep5<KN<5,2>>(long) {
}
// CHECK:      |-FunctionDecl {{.*}} prev {{.*}} skep5 'void (long)' explicit_specialization
// CHECK-NEXT: | |-TemplateArgument type 'KN<5, 2>'
// CHECK:      | |-TemplateArgument type 'long'
// CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KN<5, 2>

// FIXME: C++23 [temp.expl.spec]p12 states:
// FIXME:   ... Similarly, attributes appearing in the declaration of a template
// FIXME:   have no effect on an explicit specialization of that template.
// FIXME: Clang currently instantiates a function template specialization from
// FIXME: the function template declaration and links it as a previous
// FIXME: declaration of an explicit specialization. The instantiated
// FIXME: declaration includes attributes instantiated from the function
// FIXME: template declaration. When the instantiated declaration and the
// FIXME: explicit specialization both specify a sycl_kernel_entry_point
// FIXME: attribute with different kernel name types, a spurious diagnostic
// FIXME: is issued. The following test case is incorrectly diagnosed as
// FIXME: having conflicting kernel name types (KN<5,3> vs the incorrectly
// FIXME: inherited KN<5,-1>).
#if 0
template<>
[[clang::sycl_kernel_entry_point(KN<5,3>)]]
void skep5<KN<5,-1>>(long long) {
}
// FIXME-CHECK:      |-FunctionDecl {{.*}} prev {{.*}} skep5 'void (long long)' explicit_specialization
// FIXME-CHECK-NEXT: | |-TemplateArgument type 'KN<5, -1>'
// FIXME-CHECK:      | |-TemplateArgument type 'long long'
// FIXME-CHECK:      | `-SYCLKernelEntryPointAttr {{.*}} KN<5, 3>
#endif

template void skep5<KN<5,4>>(int);
// Checks are located with the primary template declaration above.

// Ensure that matching attributes from multiple declarations are ok.
[[clang::sycl_kernel_entry_point(KN<6>)]]
void skep6();
[[clang::sycl_kernel_entry_point(KN<6>)]]
void skep6() {
}
// CHECK:      |-FunctionDecl {{.*}} skep6 'void ()'
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<6>
// CHECK-NEXT: |-FunctionDecl {{.*}} prev {{.*}} skep6 'void ()'
// CHECK-NEXT: | |-CompoundStmt {{.*}}
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<6>

// Ensure that matching attributes from the same declaration are ok.
[[clang::sycl_kernel_entry_point(KN<7>), clang::sycl_kernel_entry_point(KN<7>)]]
void skep7() {
}
// CHECK:      |-FunctionDecl {{.*}} skep7 'void ()'
// CHECK-NEXT: | |-CompoundStmt {{.*}}
// CHECK-NEXT: | |-SYCLKernelEntryPointAttr {{.*}} KN<7>
// CHECK-NEXT: | `-SYCLKernelEntryPointAttr {{.*}} KN<7>

void the_end() {}
// CHECK:      `-FunctionDecl {{.*}} the_end 'void ()'
