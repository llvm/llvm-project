/* RUN: %clang_cc1 -std=c99 -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -verify -emit-llvm -o -  %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -verify -emit-llvm -o -  %s | FileCheck %s
 */

/* WG14 DR208: yes
 * Ambiguity in initialization
 */
int dr208_init(int);
void dr208(void) {
  int a[2] = {
    dr208_init(0),      /* expected-note {{previous initialization with side effects is here (side effects will not occur at run time)}} */
    dr208_init(1),
    [0] = dr208_init(2) /* expected-warning {{initializer overrides prior initialization of this subobject}} */
  };

  /* CHECK-NOT: call {{signext i32|i32}} @dr208_init(i32 noundef {{(signext )?}}0)
     CHECK-DAG: call {{signext i32|i32}} @dr208_init(i32 noundef {{(signext )?}}1)
     CHECK-DAG: call {{signext i32|i32}} @dr208_init(i32 noundef {{(signext )?}}2)
     CHECK-NOT: call {{signext i32|i32}} @dr208_init(i32 noundef {{(signext )?}}0)
   */
}

