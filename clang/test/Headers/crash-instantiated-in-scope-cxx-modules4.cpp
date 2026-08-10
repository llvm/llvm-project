// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -std=c++20 -x c++ -fmodule-map-file=modules.map -fmodule-name=foo1 -emit-module modules.map -o foo1.pcm
// RUN: %clang_cc1 -verify -std=c++20 -x c++ -fmodule-map-file=modules.map -fmodule-name=foo2 -emit-module modules.map -o foo2.pcm
// RUN: %clang_cc1 -verify -std=c++20 -x c++ -fmodule-map-file=modules.map -fmodule-file=foo1.pcm -fmodule-file=foo2.pcm server.cc

//--- functional
#pragma once

namespace std {
template <class> class function {};
} // namespace std

//--- foo.h
#pragma once

class MethodHandler {
 public:
  virtual ~MethodHandler() {}
  struct HandlerParameter {
    HandlerParameter();
  };
  virtual void RunHandler(const HandlerParameter &param);
};

template <class RequestType, class ResponseType>
class CallbackUnaryHandler : public MethodHandler {
 public:
  explicit CallbackUnaryHandler();

  void RunHandler(const HandlerParameter &param) final {
    void *call = nullptr;
    (void)[call](bool){};
  }
};

//--- foo1.h
// expected-no-diagnostics
#pragma once

#include "functional"

#include "foo.h"

class A;

class ClientAsyncResponseReaderHelper {
   public:
      using t = std::function<void(A)>;
        static void SetupRequest(t finish);
};

//--- foo2.h
// expected-no-diagnostics
#pragma once

#include "foo.h"

template <class BaseClass>
class a : public BaseClass {
 public:
  a() { [[maybe_unused]] CallbackUnaryHandler<int, int> a; }
};

//--- modules.map
module "foo" {
  export *
  module "foo.h" {
    export *
    textual header "foo.h"
  }
}

module "foo1" {
  export *
  module "foo1.h" {
    export *
    header "foo1.h"
  }

  use "foo"
}

module "foo2" {
  export *
  module "foo2.h" {
    export *
    header "foo2.h"
  }

  use "foo"
}

//--- server.cc
// expected-no-diagnostics
#include "functional"

#include "foo1.h"
#include "foo2.h"

std::function<void()> on_emit;

template <class RequestType, class ResponseType>
class CallbackUnaryHandler;

class s {};
class hs final : public a<s> {
  explicit hs() {}
};
