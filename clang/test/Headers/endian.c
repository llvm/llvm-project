// RUN: %clang_cc1 -triple powerpc64-unknown -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -triple powerpc64-unknown -fsyntax-only -verify -xc++ -ffreestanding %s
// RUN: %clang_cc1 -triple ppc64le-unknown -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -triple ppc64le-unknown -fsyntax-only -verify -xc++ -ffreestanding %s
// expected-no-diagnostics

#include <endian.h>


#if BYTE_ORDER == BIG_ENDIAN

_Static_assert(htobe16(0xBEEF) == 0xBEEF, "");
_Static_assert(htobe32(0xDEADBEEF) == 0xDEADBEEF, "");
_Static_assert(htobe64(0xDEADBEEFBAADF00D) == 0xDEADBEEFBAADF00D, "");
_Static_assert(htole16(0xBEEF) == 0xEFBE, "");
_Static_assert(htole32(0xDEADBEEF) == 0xEFBEADDE, "");
_Static_assert(htole64(0xDEADBEEFBAADF00D) == 0xDF0ADBAEFBEADDE, "");
_Static_assert(be16toh(0xBEEF) == 0xBEEF, "");
_Static_assert(be32toh(0xDEADBEEF) == 0xDEADBEEF, "");
_Static_assert(be64toh(0xDEADBEEFBAADF00D) == 0xDEADBEEFBAADF00D, "");
_Static_assert(le16toh(0xBEEF) == 0xEFBE, "");
_Static_assert(le32toh(0xDEADBEEF) == 0xEFBEADDE, "");
_Static_assert(le64toh(0xDEADBEEFBAADF00D) == 0xDF0ADBAEFBEADDE, "");

#elif BYTE_ORDER == LITTLE_ENDIAN

_Static_assert(htobe16(0xBEEF) == 0xEFBE, "");
_Static_assert(htobe32(0xDEADBEEF) == 0xEFBEADDE, "");
_Static_assert(htobe64(0xDEADBEEFBAADF00D) == 0xDF0ADBAEFBEADDE, "");
_Static_assert(htole16(0xBEEF) == 0xBEEF, "");
_Static_assert(htole32(0xDEADBEEF) == 0xDEADBEEF, "");
_Static_assert(htole64(0xDEADBEEFBAADF00D) == 0xDEADBEEFBAADF00D, "");
_Static_assert(be16toh(0xBEEF) == 0xEFBE, "");
_Static_assert(be32toh(0xDEADBEEF) == 0xEFBEADDE, "");
_Static_assert(be64toh(0xDEADBEEFBAADF00D) == 0xDF0ADBAEFBEADDE, "");
_Static_assert(le16toh(0xBEEF) == 0xBEEF, "");
_Static_assert(le32toh(0xDEADBEEF) == 0xDEADBEEF, "");
_Static_assert(le64toh(0xDEADBEEFBAADF00D) == 0xDEADBEEFBAADF00D, "");

#else
#error "Invalid byte order"
#endif
