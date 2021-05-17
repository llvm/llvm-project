%include <typemaps.i>

// FIXME: We need to port more typemaps from Python

//===----------------------------------------------------------------------===//

// In Lua 5.3 and beyond the VM supports integers, so we need to remap
// SWIG's internal handling of integers.


%define LLDB_NUMBER_TYPEMAP(TYPE)

// Primitive integer mapping
%typemap(in,checkfn="lua_isinteger") TYPE
%{ $1 = (TYPE)lua_tointeger(L, $input); %}
%typemap(in,checkfn="lua_isinteger") const TYPE&($basetype temp)
%{ temp=($basetype)lua_tointeger(L,$input); $1=&temp;%}
%typemap(out) TYPE
%{ lua_pushinteger(L, (lua_Integer) $1); SWIG_arg++;%}
%typemap(out) const TYPE&
%{ lua_pushinteger(L, (lua_Integer) $1); SWIG_arg++;%}

// Pointer and reference mapping
%typemap(in,checkfn="lua_isinteger") TYPE *INPUT($*ltype temp), TYPE &INPUT($*ltype temp)
%{ temp = ($*ltype)lua_tointeger(L,$input);
   $1 = &temp; %}
%typemap(in, numinputs=0) TYPE *OUTPUT ($*ltype temp)
%{ $1 = &temp; %}
%typemap(argout) TYPE *OUTPUT
%{  lua_pushinteger(L, (lua_Integer) *$1); SWIG_arg++;%}
%typemap(in) TYPE *INOUT = TYPE *INPUT;
%typemap(argout) TYPE *INOUT = TYPE *OUTPUT;
%typemap(in) TYPE &OUTPUT = TYPE *OUTPUT;
%typemap(argout) TYPE &OUTPUT = TYPE *OUTPUT;
%typemap(in) TYPE &INOUT = TYPE *INPUT;
%typemap(argout) TYPE &INOUT = TYPE *OUTPUT;
%typemap(in,checkfn="lua_isinteger") const TYPE *INPUT($*ltype temp)
%{ temp = ($*ltype)lua_tointeger(L,$input);
   $1 = &temp; %}

%enddef // LLDB_NUMBER_TYPEMAP

LLDB_NUMBER_TYPEMAP(unsigned char);
LLDB_NUMBER_TYPEMAP(signed char);
LLDB_NUMBER_TYPEMAP(short);
LLDB_NUMBER_TYPEMAP(unsigned short);
LLDB_NUMBER_TYPEMAP(signed short);
LLDB_NUMBER_TYPEMAP(int);
LLDB_NUMBER_TYPEMAP(unsigned int);
LLDB_NUMBER_TYPEMAP(signed int);
LLDB_NUMBER_TYPEMAP(long);
LLDB_NUMBER_TYPEMAP(unsigned long);
LLDB_NUMBER_TYPEMAP(signed long);
LLDB_NUMBER_TYPEMAP(long long);
LLDB_NUMBER_TYPEMAP(unsigned long long);
LLDB_NUMBER_TYPEMAP(signed long long);

%apply unsigned long { size_t };
%apply const unsigned long & { const size_t & };
%apply long { ssize_t };
%apply const long & { const ssize_t & };

//===----------------------------------------------------------------------===//

// FIXME:
//  Ideally all the typemaps should be revisited in a future SB API revision.
//  Typemaps, usually, modifies the function signatures and might spawn
//  different LLDB APIs across languages (C++, Python, Lua...).
//  Historically, typemaps have been used to replace SWIG's deficiencies,
//  but SWIG itself evolved and some API design choices are now redundant.

//===----------------------------------------------------------------------===//

// Typemap definitions to allow SWIG to properly handle char buffer.

// typemap for a char buffer
%typemap(in) (char *dst, size_t dst_len) {
   $2 = luaL_checkinteger(L, $input);
   if ($2 <= 0) {
       return luaL_error(L, "Positive integer expected");
   }
   $1 = (char *) malloc($2);
}

// SBProcess::ReadCStringFromMemory() uses a void*, but needs to be treated
// as char data instead of byte data.
%typemap(in) (void *char_buf, size_t size) = (char *dst, size_t dst_len);

// Return the char buffer.  Discarding any previous return result
%typemap(argout) (char *dst, size_t dst_len) {
   lua_pop(L, 1); // Blow away the previous result
   if ($result == 0) {
      lua_pushliteral(L, "");
   } else {
      lua_pushlstring(L, (const char *)$1, $result);
   }
   free($1);
   // SWIG_arg was already incremented
}

// SBProcess::ReadCStringFromMemory() uses a void*, but needs to be treated
// as char data instead of byte data.
%typemap(argout) (void *char_buf, size_t size) = (char *dst, size_t dst_len);

//===----------------------------------------------------------------------===//
