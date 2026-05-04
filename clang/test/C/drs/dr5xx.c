/* RUN: %clang_cc1 -std=c89 -verify=expected,c89only -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -verify=expected -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -verify=expected -pedantic %s
   RUN: %clang_cc1 -std=c17 -verify=expected -pedantic %s
   RUN: %clang_cc1 -std=c2x -verify=expected -pedantic %s
 */

/* WG14 DR502:
 * Flexible array member in an anonymous struct
 */
void dr502(void) {
  /* This is EXAMPLE 3 from 6.7.2.1 and is intended to show that a flexible
   * array member can be used when the only other members of the class are from
   * an anonymous structure member.
   */
  struct s {
    struct { int i; };
    int a[]; /* c89only-warning {{flexible array members are a C99 feature}} */
  };

  /* This is a slightly modified example that looks to see whether the
   * anonymous structure itself can provide a flexible array member for the
   * containing class.
   *
   * The committee does not think this is valid because it would mean the
   * anonymous structure would have size 0. Additionally, the anonymous
   * structure has no additional members and so the flexible array member is
   * not valid within the anonymous structure.
   */
  struct t {
    int i;
    struct { int a[]; }; /* expected-warning {{flexible array member 'a' in otherwise empty struct is a GNU extension}}
                            c89only-warning {{flexible array members are a C99 feature}}
                            expected-warning {{'' may not be nested in a struct due to flexible array member}}
                          */
  };
}

