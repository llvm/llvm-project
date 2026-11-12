// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -xc++ -emit-module -fmodule-name=internal modules.map -o internal.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -xc++ -emit-module -fmodule-name=interface modules.map -o interface.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -xc++ -emit-module -fmodule-name=foo modules.map -o foo.pcm -fmodule-file=interface.pcm -fmodule-file=internal.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-map-file=modules.map -O1 -emit-obj main.cc -verify -fmodule-file=foo.pcm

//--- modules.map
module "interface" {
  export *
  module "interface.h" {
    export *
    header "interface.h"
  }
}

module "internal" {
  export *
  module "internal.h" {
    export *
    header "internal.h"
  }
}

module "foo" {
  export *
  module "foo.h" {
    export *
    header "foo.h"
  }
}

//--- foo.h
#ifndef FOO_H
#define FOO_H

#include "strong_int.h"

DEFINE_STRONG_INT_TYPE(CallbackId, int);

#define CALL_HASH(id) \
  (void)[&]() { AbslHashValue(0, id); };

void test(CallbackId id) {
  CALL_HASH(id);
}

#include "internal.h"
#include "interface.h"

#endif

//--- interface.h
#ifndef INTERFACE_H
#define INTERFACE_H

#include "strong_int.h"

DEFINE_STRONG_INT_TYPE(EndpointToken, int);

#endif

//--- internal.h
#ifndef INTERNAL_H
#define INTERNAL_H

#include "strong_int.h"

DEFINE_STRONG_INT_TYPE(OrderedListSortKey, int);
DEFINE_STRONG_INT_TYPE(OrderedListId, int);

#endif

//--- strong_int.h
#ifndef STRONG_INT_H
#define STRONG_INT_H

namespace util_intops {

template <typename TagType, typename NativeType>
class StrongInt2;

template <typename TagType, typename NativeType>
class StrongInt2 {
 public:
  template <typename H>
  friend H AbslHashValue(H h, const StrongInt2& i) {
    return h;
  }
};

}  // namespace util_intops

#define DEFINE_STRONG_INT_TYPE(type_name, value_type)                        \
  struct type_name##_strong_int_tag_ {};                                     \
  typedef ::util_intops::StrongInt2<type_name##_strong_int_tag_, value_type> \
      type_name;

#endif

//--- main.cc
// expected-no-diagnostics
#include "foo.h"

#include "strong_int.h"

DEFINE_STRONG_INT_TYPE(ArchiveId2, int);
void partial(ArchiveId2 id) {
  CALL_HASH(id);
}
