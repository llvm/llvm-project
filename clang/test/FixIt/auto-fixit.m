/* RUN: cp %s %t
   RUN: %clang_cc1 -x objective-c -fixit %t
   RUN: %clang_cc1 -x objective-c -Werror %t
 */

// rdar://9036633

int main(void) {
  auto int i = 0;
  return i;
}
