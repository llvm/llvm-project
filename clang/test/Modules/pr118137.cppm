// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++23 %t/a.cppm -emit-llvm -o -

//--- a.h
typedef int nghttp2_session_callbacks;

//--- a.cppm
module;
#include "a.h"
export module g;
template <typename, typename T>
concept Deleter = requires(T ptr) { ptr; };
template <typename T, Deleter<T>> struct Handle {
  void GetRaw(this auto);
};
struct SessionCallbacksDeleter
    : Handle<nghttp2_session_callbacks, SessionCallbacksDeleter> {
} Server_callbacks;
void Server() { Server_callbacks.GetRaw(); }
