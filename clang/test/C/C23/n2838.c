// RUN: %clang_cc1 -triple x86_64 -verify -std=c2x %s

/* WG14 N2838: yes
 * Types and sizes
 */

char buffer4[0xFFFF'FFFF'FFFF'FFFF'1wb]; /* expected-error {{array is too large (295147905179352825841 elements)}} */
char buffer3[0xFFFF'FFFF'FFFF'FFFFwb];   /* expected-error {{array is too large (18446744073709551615 elements)}} */
char buffer2[0x7FFF'FFFF'FFFF'FFFFwb];   /* expected-error {{array is too large (9223372036854775807 elements)}} */
char buffer1[0x1FFF'FFFF'FFFF'FFFFwb];   /* array is juuuuuust right */

/* The largest object we can create is still smaller than SIZE_MAX. */
static_assert(0x1FFF'FFFF'FFFF'FFFFwb <= __SIZE_MAX__);
