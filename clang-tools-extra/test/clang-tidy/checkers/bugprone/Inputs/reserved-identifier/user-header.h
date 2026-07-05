#define _HEADER_MACRO(m) int m = 0

namespace _Header_Ns {
class _Header_Object {
  int _Header_Member;
};

float _Header_Global;

void _Header_Function() {}

using _Header_Alias = int;
} // namespace _Header_Ns

//

#define __header_macro(m) int m = 0

namespace __header_ns {
class __header_object {
  int __header_member;
};

float __header_global;

void __header_function() {}

using __header_alias = int;
} // namespace __header_ns

//

#define header_macro__m(m) int m = 0

namespace header_ns__n {
class header_object__o {
  int header_member__m;
};

float header_global__g;

void header_function__f() {}

using header_alias__a = int;
} // namespace header_ns__n

//

#define _header_macro(m) int m = 0

namespace _header_ns {}
class _header_object {};

float _header_global;

void _header_function() {}

using _header_alias = int;
