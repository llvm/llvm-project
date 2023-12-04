// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.ClassCase: Camel_Snake_Case, \
// RUN:     readability-identifier-naming.StructCase: camel_Snake_Back, \
// RUN:   }}'

// clang-format off

//===----------------------------------------------------------------------===//
// Camel_Snake_Case tests
//===----------------------------------------------------------------------===//
class XML_Parser {};
class Xml_Parser {};
class XML_Parser_2 {};
// NO warnings or fixes expected as these identifiers are Camel_Snake_Case

class XmlParser {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'XmlParser'
// CHECK-FIXES: {{^}}class Xml_Parser {};{{$}}

class Xml_parser {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'Xml_parser'
// CHECK-FIXES: {{^}}class Xml_Parser {};{{$}}

class xml_parser {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'xml_parser'
// CHECK-FIXES: {{^}}class Xml_Parser {};{{$}}

class xml_Parser {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'xml_Parser'
// CHECK-FIXES: {{^}}class Xml_Parser {};{{$}}

class xml_Parser_2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'xml_Parser_2'
// CHECK-FIXES: {{^}}class Xml_Parser_2 {};{{$}}

class t {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 't'
// CHECK-FIXES: {{^}}class T {};{{$}}

//===----------------------------------------------------------------------===//
// camel_Snake_Back tests
//===----------------------------------------------------------------------===//
struct json_Parser {};
struct json_Parser_2 {};
struct u {};
// NO warnings or fixes expected as these identifiers are camel_Snake_Back

struct JsonParser {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'JsonParser'
// CHECK-FIXES: {{^}}struct json_Parser {};{{$}}

struct Json_parser {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'Json_parser'
// CHECK-FIXES: {{^}}struct json_Parser {};{{$}}

struct json_parser {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'json_parser'
// CHECK-FIXES: {{^}}struct json_Parser {};{{$}}

