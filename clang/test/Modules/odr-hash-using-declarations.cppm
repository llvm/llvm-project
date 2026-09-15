// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++20 -fno-skip-odr-check-in-gmf -emit-module-interface %t/part.cppm -o %t/part.pcm
// RUN: %clang_cc1 -std=c++20 -fno-skip-odr-check-in-gmf -fmodule-file=repro:part=%t/part.pcm -emit-module-interface %t/module.cppm -o %t/module.pcm -verify
// RUN: %clang_cc1 -std=c++20 -DREVERSE -fno-skip-odr-check-in-gmf -emit-module-interface %t/part.cppm -o %t/part.pcm
// RUN: %clang_cc1 -std=c++20 -DREVERSE -fno-skip-odr-check-in-gmf -fmodule-file=repro:part=%t/part.pcm -emit-module-interface %t/module.cppm -o %t/module.pcm -verify
// RUN: %clang_cc1 -std=c++20 -fno-skip-odr-check-in-gmf -emit-module-interface %t/first.cppm -o %t/first.pcm
// RUN: %clang_cc1 -std=c++20 -fno-skip-odr-check-in-gmf -emit-module-interface %t/second.cppm -o %t/second.pcm
// RUN: not %clang_cc1 -std=c++20 -fno-skip-odr-check-in-gmf -fprebuilt-module-path=%t -fsyntax-only %t/bad.cpp 2>&1 | FileCheck %s --check-prefix=BAD
// BAD-DAG: error: 'different_spelling::qualification' has different definitions
// BAD-DAG: error: 'different_spelling::keyword' has different definitions
// BAD-DAG: error: 'different_spelling::cv' has different definitions

//--- declarations.h
typedef long T;
using Qualified = const volatile long;
struct Record { int value; };
enum class Kind { value };
typedef struct CRecord { int value; } CRecord;

namespace chain {
using ::T;
}

namespace ns {
#ifdef INDIRECT
using chain::T;
using ::Qualified;
using ::Record;
using ::Kind;
using ::CRecord;
#endif

inline void fun() { (void)(T)0; }
inline Qualified *pointer(Qualified *p) { return p; }
inline Record make() { return Record{0}; }
inline unsigned sizes() {
  return sizeof(T) + sizeof(Record) + sizeof(Kind) + sizeof(CRecord);
}
struct Fields {
  T value;
  Qualified *pointer;
  Record record;
  Kind kind;
};
template <class U> inline void templ(U) {
  const T value = 0;
  (void)value;
}
}

//--- part.cppm
module;
#ifdef REVERSE
#define INDIRECT
#endif
#include "declarations.h"
export module repro:part;
export void use() {
  ns::Fields fields{};
  ns::fun();
  ns::pointer(nullptr);
  ns::make();
  ns::sizes();
  ns::templ(0);
}

//--- module.cppm
// expected-no-diagnostics
module;
#ifndef REVERSE
#define INDIRECT
#endif
#include "declarations.h"
export module repro;
export import :part;

//--- bad.h
namespace types {
using Scalar = int;
struct Record {};
}
namespace different_spelling {
using types::Scalar;
using types::Record;
inline int qualification() {
#ifdef FIRST
  return sizeof(Scalar);
#else
  return sizeof(different_spelling::Scalar);
#endif
}
inline int keyword() {
#ifdef FIRST
  return sizeof(Record);
#else
  return sizeof(struct Record);
#endif
}
inline int cv() {
#ifdef FIRST
  return sizeof(Scalar);
#else
  return sizeof(const Scalar);
#endif
}
}

//--- first.cppm
module;
#define FIRST
#include "bad.h"
export module first;
export using different_spelling::qualification;
export using different_spelling::keyword;
export using different_spelling::cv;

//--- second.cppm
module;
#include "bad.h"
export module second;
export using different_spelling::qualification;
export using different_spelling::keyword;
export using different_spelling::cv;

//--- bad.cpp
import first;
import second;
int use() { return qualification() + keyword() + cv(); }
