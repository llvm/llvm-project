/* RUN: %clang_cc1 -std=c89 -Wno-deprecated-non-prototype -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -Wno-deprecated-non-prototype -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -Wno-deprecated-non-prototype -ast-dump -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -Wno-deprecated-non-prototype -ast-dump -o -  %s | FileCheck %s
 */

/* WG14 DR206: yes
 * Default argument conversion of float _Complex
 */
void dr206_unprototyped();
void dr206(void) {
  /* Ensure that _Complex float is not promoted to _Complex double but is
   * instead passed directly without a type conversion.
   */
  _Complex float f = 1.2f;
  dr206_unprototyped(f);
  // CHECK: CallExpr 0x{{.*}} <line:16:3, col:23> 'void'
  // CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:3> 'void (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:3> 'void ()' Function 0x{{.*}} 'dr206_unprototyped' 'void ()'
  // CHECK-NEXT: ImplicitCastExpr 0x{{.*}} <col:22> '_Complex float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr 0x{{.*}} <col:22> '_Complex float' lvalue Var 0x{{.*}} 'f' '_Complex float'
}

