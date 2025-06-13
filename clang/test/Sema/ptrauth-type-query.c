// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -std=c23 -verify -fptrauth-calls -fptrauth-intrinsics %s

#include <ptrauth.h>

struct __attribute__((ptrauth_struct(3, 100))) PtrauthStruct {
  void *ptr;
};

struct NoPtrauthStruct {
  void *ptr;
};

void test_func(void) {
   int *no_auth = 0;
   _Static_assert(!__builtin_ptrauth_has_authentication(typeof(no_auth)));
   __builtin_ptrauth_schema_key(typeof(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_key parameter is not an authenticated value}}
   __builtin_ptrauth_schema_is_address_discriminated(typeof(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_is_address_discriminated parameter is not an authenticated value}}
   __builtin_ptrauth_schema_extra_discriminator(typeof(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_extra_discriminator parameter is not an authenticated value}}
   __builtin_ptrauth_schema_options(typeof(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_options parameter is not an authenticated value}}


   int *__ptrauth(1,0,1) no_addr_disc = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(no_addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(no_addr_disc)) == 1);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(typeof(no_addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(no_addr_disc)) == 1);
   int *__ptrauth(1,1,2) addr_disc = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(addr_disc)) == 1);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(typeof(addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(addr_disc)) == 2);
   int *__ptrauth(2,1,3) key2 = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(key2)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(key2)) == 2);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(typeof(key2)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(key2)) == 3);

   int *__ptrauth(1, 0, 4, "strip") strip = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(strip)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(strip)) == 1);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(typeof(strip)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(strip)) == 4);
   _Static_assert(__builtin_strcmp("strip", __builtin_ptrauth_schema_options(typeof(strip))) == 0);

   int *__ptrauth(1, 0, 4, __builtin_ptrauth_schema_options(typeof(strip))) strip2 = 0;
   __auto_type test_ptr = &strip;
   // Verify matching pointer auth schema
   test_ptr = &strip2;

   void (*normal_func_ptr)(int) = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(normal_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(normal_func_ptr)) == ptrauth_key_function_pointer);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(typeof(normal_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(normal_func_ptr)) == 0);

   void (* __ptrauth(1,1,5) explicit_signed_func_ptr)(int)  = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(explicit_signed_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_key(typeof(explicit_signed_func_ptr)) == 1);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(typeof(explicit_signed_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(explicit_signed_func_ptr)) == 5);

   _Static_assert(!__builtin_ptrauth_has_authentication(float));

   struct PtrauthStruct *S;
   _Static_assert(__builtin_ptrauth_has_authentication(typeof(S)));
   _Static_assert(__builtin_ptrauth_has_authentication(struct PtrauthStruct *));
   _Static_assert(__builtin_ptrauth_schema_key(struct PtrauthStruct *) == 3);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(struct PtrauthStruct *) == 0);
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(struct PtrauthStruct *) == 100);
   _Static_assert(__builtin_strcmp("", __builtin_ptrauth_schema_options(struct PtrauthStruct *)) == 0);
   struct PtrauthStruct * __ptrauth(1, 1, 101, "strip") overrideAuth;
   _Static_assert(__builtin_ptrauth_schema_key(typeof(overrideAuth)) == 1);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(typeof(overrideAuth)) == 1);
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(typeof(overrideAuth)) == 101);
   _Static_assert(__builtin_strcmp("strip", __builtin_ptrauth_schema_options(typeof(overrideAuth))) == 0);

   _Static_assert(__builtin_ptrauth_schema_key(struct NoPtrauthStruct *) == 1);
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_key parameter is not an authenticated value}}
}
