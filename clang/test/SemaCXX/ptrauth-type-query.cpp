// RUN: %clang_cc1 -triple arm64-apple-ios -fsyntax-only -std=c++2b -verify -fptrauth-calls -fptrauth-intrinsics %s

#include <ptrauth.h>

void test_func(void) {
   int *no_auth = 0;
   _Static_assert(!__builtin_ptrauth_has_authentication(decltype(no_auth)));
   __builtin_ptrauth_schema_key(decltype(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_key parameter is not an authenticated value}}
   __builtin_ptrauth_schema_is_address_discriminated(decltype(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_is_address_discriminated parameter is not an authenticated value}}
   __builtin_ptrauth_schema_extra_discriminator(decltype(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_extra_discriminator parameter is not an authenticated value}}
   __builtin_ptrauth_schema_options(decltype(no_auth));
   // expected-error@-1 {{argument to __builtin_ptrauth_schema_options parameter is not an authenticated value}}


   int *__ptrauth(1,0,1) no_addr_disc = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(no_addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(no_addr_disc)) == 1);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(decltype(no_addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(no_addr_disc)) == 1);
   int *__ptrauth(1,1,2) addr_disc = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(addr_disc)) == 1);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(decltype(addr_disc)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(addr_disc)) == 2);
   int *__ptrauth(2,1,3) key2 = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(key2)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(key2)) == 2);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(decltype(key2)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(key2)) == 3);

   int *__ptrauth(1, 0, 4, "strip") strip = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(strip)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(strip)) == 1);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(decltype(strip)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(strip)) == 4);
   _Static_assert(__builtin_strcmp("strip", __builtin_ptrauth_schema_options(decltype(strip))) == 0);

   int *__ptrauth(1, 0, 4, __builtin_ptrauth_schema_options(decltype(strip))) strip2 = 0;
   __auto_type test_ptr = &strip;
   // Verify matching pointer auth schema
   test_ptr = &strip2;

   int *__ptrauth(1,0,5, "authenticates-null-values,sign-and-strip,isa-pointer") multi_option;
   _Static_assert(__builtin_strcmp("sign-and-strip,isa-pointer,authenticates-null-values", __builtin_ptrauth_schema_options(decltype(multi_option))) == 0);
   int *__ptrauth(1,0,5, "sign-and-strip,isa-pointer,authenticates-null-values") multi_option2;
   _Static_assert(__builtin_strcmp(__builtin_ptrauth_schema_options(decltype(multi_option2)), __builtin_ptrauth_schema_options(decltype(multi_option))) == 0);


   int *__ptrauth(1,0,5, "sign-and-auth") default_options;
   _Static_assert(__builtin_strcmp("", __builtin_ptrauth_schema_options(decltype(default_options))) == 0);

   void (*normal_func_ptr)(int) = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(normal_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(normal_func_ptr)) == ptrauth_key_function_pointer);
   _Static_assert(!__builtin_ptrauth_schema_is_address_discriminated(decltype(normal_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(normal_func_ptr)) == 0);
   typedef void(*function_type)(int);
   function_type __ptrauth(1,1,5) explicit_signed_func_ptr = 0;
   _Static_assert(__builtin_ptrauth_has_authentication(decltype(explicit_signed_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_key(decltype(explicit_signed_func_ptr)) == 1);
   _Static_assert(__builtin_ptrauth_schema_is_address_discriminated(decltype(explicit_signed_func_ptr)));
   _Static_assert(__builtin_ptrauth_schema_extra_discriminator(decltype(explicit_signed_func_ptr)) == 5);

}

template <typename T, bool has_ptrauth=__builtin_ptrauth_has_authentication(T)> struct PtrauthExtractor;
template <typename T> struct PtrauthExtractor<T, false> {
   static const bool isAuthenticated = false;
   static const int key = -1;
   static const int isAddressDiscriminated = false;
   static const int extraDiscriminator = -1;
   constexpr static auto options = "no-options";
};

template <typename T> struct PtrauthExtractor<T, true> {
   static const bool isAuthenticated = true;
   static const int key = __builtin_ptrauth_schema_key(T);
   static const int isAddressDiscriminated = __builtin_ptrauth_schema_is_address_discriminated(T);
   static const int extraDiscriminator = __builtin_ptrauth_schema_extra_discriminator(T);
   constexpr static auto options = __builtin_ptrauth_schema_options(T);
};

template <typename T> void template_test() {
   typedef PtrauthExtractor<T> TestStruct;
   static_assert(__builtin_ptrauth_has_authentication(T) == TestStruct::isAuthenticated);
   static_assert(__builtin_ptrauth_schema_key(T) == TestStruct::key);
   // expected-error@-1 2 {{argument to __builtin_ptrauth_schema_key parameter is not an authenticated value}}
   static_assert(__builtin_ptrauth_schema_is_address_discriminated(T) == TestStruct::isAddressDiscriminated);
   static_assert(__builtin_ptrauth_schema_extra_discriminator(T) == TestStruct::extraDiscriminator);
   static_assert(__builtin_strcmp(__builtin_ptrauth_schema_options(T), TestStruct::options) == 0);
}

void test_template_instantiator() {
   template_test<int*>();
   //expected-note@-1 {{in instantiation of function template specialization 'template_test<int *>' requested here}}
   template_test<int* __ptrauth(1,1,1,"strip")>();
   template_test<int* __ptrauth(2,1,2,"")>();
   template_test<int* __ptrauth(3,0,3,"isa-pointer")>();
   template_test<float>();
   // expected-note@-1 {{in instantiation of function template specialization 'template_test<float>' requested here}}
}

