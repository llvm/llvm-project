// RUN: not %clang_cc1 -std=c++2c -ast-dump %s | FileCheck  %s

namespace InvalidPacksShouldNotCrash {

struct NotAPack;
template <typename T, auto V, template<typename> typename Tp>
void not_pack() {
    int i = 0;
    i...[0]; // expected-error {{i does not refer to the name of a parameter pack}}
    V...[0]; // expected-error {{V does not refer to the name of a parameter pack}}
    NotAPack...[0] a; // expected-error{{'NotAPack' does not refer to the name of a parameter pack}}
    T...[0] b;   // expected-error{{'T' does not refer to the name of a parameter pack}}
    Tp...[0] c; // expected-error{{'Tp' does not refer to the name of a parameter pack}}
}

// CHECK:      FunctionDecl {{.*}} not_pack 'void ()'
// CHECK:           DeclStmt {{.*}}
// CHECK:           DeclStmt {{.*}}
// CHECK-NEXT:        VarDecl {{.*}} a 'NotAPack...{{.*}}'
// CHECK-NEXT:      DeclStmt {{.*}}
// CHECK-NEXT:        VarDecl {{.*}} 'T...{{.*}}'
// CHECK-NEXT:       DeclStmt {{.*}}
// CHECK-NEXT:        VarDecl {{.*}} c 'Tp...{{.*}}'

}
