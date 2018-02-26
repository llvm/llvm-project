// RUN: %clang_cc1 -std=c++17 -fsyntax-only -fmodules %s

#pragma clang module build a
module a {}
#pragma clang module contents
#pragma clang module begin a

template <class c, c e> struct ak { static constexpr c value = e; };
ak<bool, true> instantiate_class_definition;

#pragma clang module end /* a */
#pragma clang module endbuild


#pragma clang module build o
module o {}
#pragma clang module contents
#pragma clang module begin o
#pragma clang module import a

inline int instantiate_var_definition() { return ak<bool, true>::value; }

#pragma clang module end
#pragma clang module endbuild


#pragma clang module import o
#pragma clang module import a

int main() { return ak<bool, true>::value; }
