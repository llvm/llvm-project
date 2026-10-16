// RUN: %clang_cc1 -triple x86_64 -verify -std=c2x %s

/* WG14 N2838: yes
 * Types and sizes
 */

char buffer4[0xFFFF'FFFF'FFFF'FFFF'1wb]; /* expected-error {{array is too large (295'147'905'179'352'825'841 elements)}} */
char buffer3[0xFFFF'FFFF'FFFF'FFFFwb];   /* expected-error {{array is too large (18'446'744'073'709'551'615 elements)}} */
char buffer2[0x7FFF'FFFF'FFFF'FFFFwb];   /* expected-error {{array is too large (9'223'372'036'854'775'807 elements)}} */
char buffer1[0x1FFF'FFFF'FFFF'FFFFwb];   /* array is juuuuuust right */

/* The largest object we can create is still smaller than SIZE_MAX. */
static_assert(0x1FFF'FFFF'FFFF'FFFFwb <= __SIZE_MAX__);
