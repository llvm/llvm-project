/*===---- ptrauth.h - Pointer authentication -------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __PTRAUTH_H
#define __PTRAUTH_H

typedef enum {
  ptrauth_key_asia = 0,
  ptrauth_key_asib = 1,
  ptrauth_key_asda = 2,
  ptrauth_key_asdb = 3,
} ptrauth_key;

/* An integer type of the appropriate size for a discriminator argument. */
typedef __UINTPTR_TYPE__ ptrauth_extra_data_t;

/* An integer type of the appropriate size for a generic signature. */
typedef __UINTPTR_TYPE__ ptrauth_generic_signature_t;

/* A signed pointer value embeds the original pointer together with
   a signature that attests to the validity of that pointer.  Because
   this signature must use only "spare" bits of the pointer, a
   signature's validity is probabilistic in practice: it is unlikely
   but still plausible that an invalidly-derived signature will
   somehow equal the correct signature and therefore successfully
   authenticate.  Nonetheless, this scheme provides a strong degree
   of protection against certain kinds of attacks. */

/* Authenticating a pointer that was not signed with the given key
   and extra-data value will (likely) fail by trapping. */

#if __has_feature(ptrauth_intrinsics)

/* Strip the signature from a value without authenticating it.

   If the value is a function pointer, the result will not be a
   legal function pointer because of the missing signature, and
   attempting to call it will result in an authentication failure.

   The value must be an expression of pointer type.
   The key must be a constant expression of type ptrauth_key.
   The result will have the same type as the original value. */
#define ptrauth_strip(__value, __key) __builtin_ptrauth_strip(__value, __key)

/* Blend a constant discriminator into the given pointer-like value
   to form a new discriminator.  Not all bits of the inputs are
   guaranteed to contribute to the result.

   On arm64e, the integer must fall within the range of a uint16_t;
   other bits may be ignored.

   The first argument must be an expression of pointer type.
   The second argument must be an expression of integer type.
   The result will have type uintptr_t. */
#define ptrauth_blend_discriminator(__pointer, __integer)                      \
  __builtin_ptrauth_blend_discriminator(__pointer, __integer)

/* Add a signature to the given pointer value using a specific key,
   using the given extra data as a salt to the signing process.

   This operation does not authenticate the original value and is
   therefore potentially insecure if an attacker could possibly
   control that value.

   The value must be an expression of pointer type.
   The key must be a constant expression of type ptrauth_key.
   The extra data must be an expression of pointer or integer type;
   if an integer, it will be coerced to ptrauth_extra_data_t.
   The result will have the same type as the original value. */
#define ptrauth_sign_unauthenticated(__value, __key, __data)                   \
  __builtin_ptrauth_sign_unauthenticated(__value, __key, __data)

/* Authenticate a pointer using one scheme and resign it using another.

   If the result is subsequently authenticated using the new scheme, that
   authentication is guaranteed to fail if and only if the initial
   authentication failed.

   The value must be an expression of pointer type.
   The key must be a constant expression of type ptrauth_key.
   The extra data must be an expression of pointer or integer type;
   if an integer, it will be coerced to ptrauth_extra_data_t.
   The result will have the same type as the original value.

   This operation is guaranteed to not leave the intermediate value
   available for attack before it is re-signed.

   Do not pass a null pointer to this function. A null pointer
   will not successfully authenticate.

   This operation traps if the authentication fails. */
#define ptrauth_auth_and_resign(__value, __old_key, __old_data, __new_key,     \
                                __new_data)                                    \
  __builtin_ptrauth_auth_and_resign(__value, __old_key, __old_data, __new_key, \
                                    __new_data)

/* Authenticate a data pointer.

   The value must be an expression of non-function pointer type.
   The key must be a constant expression of type ptrauth_key.
   The extra data must be an expression of pointer or integer type;
   if an integer, it will be coerced to ptrauth_extra_data_t.
   The result will have the same type as the original value.

   This operation traps if the authentication fails. */
#define ptrauth_auth_data(__value, __old_key, __old_data)                      \
  __builtin_ptrauth_auth(__value, __old_key, __old_data)

/* Compute a signature for the given pair of pointer-sized values.
   The order of the arguments is significant.

   Like a pointer signature, the resulting signature depends on
   private key data and therefore should not be reliably reproducible
   by attackers.  That means that this can be used to validate the
   integrity of arbitrary data by storing a signature for that data
   alongside it, then checking that the signature is still valid later.
   Data which exceeds two pointers in size can be signed by either
   computing a tree of generic signatures or just signing an ordinary
   cryptographic hash of the data.

   The result has type ptrauth_generic_signature_t.  However, it may
   not have as many bits of entropy as that type's width would suggest;
   some implementations are known to compute a compressed signature as
   if the arguments were a pointer and a discriminator.

   The arguments must be either pointers or integers; if integers, they
   will be coerce to uintptr_t. */
#define ptrauth_sign_generic_data(__value, __data)                             \
  __builtin_ptrauth_sign_generic_data(__value, __data)

#else

#define ptrauth_strip(__value, __key)                                          \
  ({                                                                           \
    (void)__key;                                                               \
    __value;                                                                   \
  })

#define ptrauth_blend_discriminator(__pointer, __integer)                      \
  ({                                                                           \
    (void)__pointer;                                                           \
    (void)__integer;                                                           \
    ((ptrauth_extra_data_t)0);                                                 \
  })

#define ptrauth_sign_unauthenticated(__value, __key, __data)                   \
  ({                                                                           \
    (void)__key;                                                               \
    (void)__data;                                                              \
    __value;                                                                   \
  })

#define ptrauth_auth_and_resign(__value, __old_key, __old_data, __new_key,     \
                                __new_data)                                    \
  ({                                                                           \
    (void)__old_key;                                                           \
    (void)__old_data;                                                          \
    (void)__new_key;                                                           \
    (void)__new_data;                                                          \
    __value;                                                                   \
  })

#define ptrauth_auth_data(__value, __old_key, __old_data)                      \
  ({                                                                           \
    (void)__old_key;                                                           \
    (void)__old_data;                                                          \
    __value;                                                                   \
  })

#define ptrauth_sign_generic_data(__value, __data)                             \
  ({                                                                           \
    (void)__value;                                                             \
    (void)__data;                                                              \
    ((ptrauth_generic_signature_t)0);                                          \
  })

#endif /* __has_feature(ptrauth_intrinsics) */

#endif /* __PTRAUTH_H */
