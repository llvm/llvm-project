// XFAIL: linux

// RUN: rm -rf %t
// RUN: %clang_cc1 %s -index-store-path %t/idx -D THE_TYPE=long
// RUN: %clang_cc1 %s -index-store-path %t/idx -D THE_TYPE=char
// RUN: find %t/idx/*/records -name "record-hash*" | count 2

template<typename T>
class TC {};

// This should result in different records, due to the different template parameter type.
void some_func(TC<THE_TYPE>);
