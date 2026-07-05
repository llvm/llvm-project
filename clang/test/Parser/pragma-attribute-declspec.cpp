// RUN: %clang_cc1 -triple i386-pc-win32 -std=c++11  -verify -Wno-pragma-clang-attribute -fms-extensions -fms-compatibility %s

#pragma clang attribute push(__declspec(dllexport), apply_to = function)

void function();

#pragma clang attribute pop

#pragma clang attribute push(__declspec(dllexport, dllimport), apply_to = function)
#pragma clang attribute pop

#pragma clang attribute push(__declspec(align), apply_to = variable) // expected-error {{attribute 'align' is not supported by '#pragma clang attribute'}}

#pragma clang attribute push(__declspec(), apply_to = variable) // A noop
