// RUN: %clang_cc1 -x c++ -verify %s 

// dr2354: 15

namespace DR2354 {

enum alignas(64) A {};        // expected-error {{'alignas' attribute cannot be applied to an enumeration}}
enum struct alignas(64) B {}; // expected-error {{'alignas' attribute cannot be applied to an enumeration}}

} // namespace DR2354
