// RUN: %check_clang_tidy %s bugprone-reserved-identifier %t -- -- \
// RUN:   -I%S/Inputs/reserved-identifier \
// RUN:   -isystem %S/Inputs/reserved-identifier/system

// no warnings expected without -header-filter=
#include "user-header.h"
#include <system-header.h>

#define _MACRO(m) int m = 0
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: declaration uses identifier '_MACRO', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}#define MACRO(m) int m = 0{{$}}

namespace _Ns {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration uses identifier '_Ns', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}namespace Ns {{{$}}

class _Object {
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_Object', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}class Object {{{$}}
  int _Member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_Member', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}  int Member;{{$}}
};

float _Global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_Global', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}float Global;{{$}}

void _Function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '_Function', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void Function() {}{{$}}

using _Alias = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_Alias', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}using Alias = int;{{$}}

template <typename _TemplateParam>
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: declaration uses identifier '_TemplateParam', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}template <typename TemplateParam>{{$}}
struct S {};

} // namespace _Ns

//

#define __macro(m) int m = 0
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: declaration uses identifier '__macro', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}#define _macro(m) int m = 0{{$}}

namespace __ns {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration uses identifier '__ns', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}namespace ns {{{$}}
class __object {
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '__object', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}class _object {{{$}}
  int __member;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '__member', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}  int _member;{{$}}
};

float __global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '__global', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}float _global;{{$}}

void __function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '__function', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void _function() {}{{$}}

using __alias = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '__alias', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}using _alias = int;{{$}}

template <typename __templateParam>
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: declaration uses identifier '__templateParam', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}template <typename _templateParam>{{$}}
struct S {};

} // namespace __ns

//

#define macro___m(m) int m = 0
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: declaration uses identifier 'macro___m', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}#define macro_m(m) int m = 0{{$}}

namespace ns___n {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration uses identifier 'ns___n', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}namespace ns_n {{{$}}
class object___o {
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier 'object___o', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}class object_o {{{$}}
  int member___m;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier 'member___m', which is a reserved identifier [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}  int member_m;{{$}}
};

float global___g;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier 'global___g', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}float global_g;{{$}}

void function___f() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier 'function___f', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void function_f() {}{{$}}

using alias___a = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier 'alias___a', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}using alias_a = int;{{$}}

template <typename templateParam___t>
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: declaration uses identifier 'templateParam___t', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}template <typename templateParam_t>{{$}}
struct S {};

} // namespace ns___n

//

// OK, this rule does not apply to macros:
// https://github.com/llvm/llvm-project/issues/64130#issuecomment-1655751676
#define _macro(m) int m = 0

namespace _ns {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: declaration uses identifier '_ns', which is reserved in the global namespace [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}namespace ns {{{$}}
int _i;
// no warning
} // namespace _ns
class _object {
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_object', which is reserved in the global namespace [bugprone-reserved-identifier]
  // CHECK-FIXES: {{^}}class object {{{$}}
  int _member;
  // no warning
};
float _global;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_global', which is reserved in the global namespace [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}float global;{{$}}
void _function() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '_function', which is reserved in the global namespace [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void function() {}{{$}}
using _alias = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: declaration uses identifier '_alias', which is reserved in the global namespace [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}using alias = int;{{$}}
template <typename _templateParam> // no warning, template params are not in the global namespace
struct S {};

void _float() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '_float', which is reserved in the global namespace; cannot be fixed because 'float' would conflict with a keyword [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void _float() {}{{$}}

#define SOME_MACRO
int SOME__MACRO;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: declaration uses identifier 'SOME__MACRO', which is a reserved identifier; cannot be fixed because 'SOME_MACRO' would conflict with a macro definition [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}int SOME__MACRO;{{$}}

void _TWO__PROBLEMS() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '_TWO__PROBLEMS', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void TWO_PROBLEMS() {}{{$}}
void _two__problems() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: declaration uses identifier '_two__problems', which is a reserved identifier [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}void two_problems() {}{{$}}

int __;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: declaration uses identifier '__', which is a reserved identifier; cannot be fixed automatically [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}int __;{{$}}

int _________;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: declaration uses identifier '_________', which is a reserved identifier; cannot be fixed automatically [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}int _________;{{$}}

int _;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: declaration uses identifier '_', which is reserved in the global namespace; cannot be fixed automatically [bugprone-reserved-identifier]
// CHECK-FIXES: {{^}}int _;{{$}}

// This should not trigger a warning
// https://github.com/llvm/llvm-project/issues/52895
#define _5_kmph_rpm 459

// these should pass
#define MACRO(m) int m = 0

namespace Ns {
class Object {
  int Member;
};
float Global;

void Function() {}
using Alias = int;
template <typename TemplateParam>
struct S {};
} // namespace Ns
namespace ns_ {
class object_ {
  int member_;
};
float global_;
void function_() {}
using alias_ = int;
template <typename templateParam_>
struct S {};
} // namespace ns_

class object_ {
  int member_;
};
float global_;
void function_() {}
using alias_ = int;
template <typename templateParam_>
struct S_ {};
