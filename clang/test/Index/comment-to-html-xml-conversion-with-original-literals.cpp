// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -std=c++11 | FileCheck %s

constexpr int value(float f) { return int(f); }

enum MyEnum {
hexadecimal = 0x10 //!< a
// CHECK: <Declaration>hexadecimal = 0x10</Declaration>

, withSuffix = 1u + 010 //!< b
// CHECK: <Declaration>withSuffix = 1u + 010</Declaration>

#define ARG(x) x
, macroArg = ARG(0x1) //!< c
// CHECK: <Declaration>macroArg = ARG(0x1)</Declaration>

#define MACROCONCAT(x, y) 22##x##y
, macroConcat = MACROCONCAT(3, 2) //!< d
// CHECK: <Declaration>macroConcat = MACROCONCAT(3, 2)</Declaration>

#define MACRO(a,n) = 0x##a##n
, weirdMacros MACRO(2,1) //!< e
// CHECK: <Declaration>weirdMacros = 33</Declaration>

, floatLiteral = value(0.25e3) //!< f
// CHECK: <Declaration>floatLiteral = value(0.25e3)</Declaration>
};
