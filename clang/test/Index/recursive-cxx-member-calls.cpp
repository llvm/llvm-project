typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
namespace std {
  template < class _T1, class _T2 > struct pair { _T2 second; };
}
extern "C" {
  int memcmp(const void *, const void *, size_t);
  size_t strlen(const char *);
}
namespace       clang {
  class IdentifierInfo;
  class AttributeList {
    enum Kind {
      AT_IBAction, AT_IBOutlet, AT_IBOutletCollection,
      AT_address_space, AT_alias, AT_aligned, AT_always_inline,
      AT_analyzer_noreturn, AT_annotate, AT_base_check, AT_blocks,
      AT_carries_dependency, AT_cdecl, AT_cleanup, AT_const, AT_constructor,
      AT_deprecated, AT_destructor, AT_dllexport, AT_dllimport,
      AT_ext_vector_type, AT_fastcall, AT_final, AT_format, AT_format_arg,
      AT_gnu_inline, AT_hiding, AT_malloc, AT_mode, AT_naked, AT_nodebug,
      AT_noinline, AT_no_instrument_function, AT_nonnull, AT_noreturn,
      AT_nothrow, AT_nsobject, AT_objc_exception, AT_override,
      AT_cf_returns_not_retained, AT_cf_returns_retained,
      AT_ns_returns_not_retained, AT_ns_returns_retained, AT_objc_gc, 
      AT_overloadable, AT_ownership_holds, AT_ownership_returns,
      AT_ownership_takes, AT_packed, AT_pascal, AT_pure, AT_regparm,
      AT_section, AT_sentinel, AT_stdcall, AT_thiscall, AT_transparent_union,
      AT_unavailable, AT_unused, AT_used, AT_vecreturn, AT_vector_size,
      AT_visibility, AT_warn_unused_result, AT_weak, AT_weakref,
      AT_weak_import, AT_reqd_wg_size, AT_init_priority,
      AT_returns_twice, IgnoredAttribute, UnknownAttribute
    };
    static Kind getKind(const IdentifierInfo * Name);
  };
}
size_t magic_length(const char *s);
namespace llvm {
class StringRef {
public:
  typedef const char *iterator;
  static const size_t npos = ~size_t(0);
private:
  const char *Data;
  size_t Length;
  static size_t min(size_t a, size_t b) { return a < b ? a : b; }
public:
  StringRef(): Data(0), Length(0) {}
  StringRef(const char *Str) : Data(Str), Length(magic_length(Str)) {}
  StringRef(const char *data, size_t length) : Data(data), Length(length) {}
  iterator end() const { return Data; }
  size_t size() const { return Length; }
  bool startswith(StringRef Prefix) const {
    return Length >= Prefix.Length &&
          memcmp(Data, Prefix.Data, Prefix.Length) == 0;
  }
  bool endswith(StringRef Suffix) const {
    return Length >= Suffix.Length &&
      memcmp(end() - Suffix.Length, Suffix.Data, Suffix.Length) == 0;
  }
  StringRef substr(size_t Start, size_t N = npos) const {
    return StringRef(Data + Start, min(N, Length - Start));
  }
};
}
namespace clang {
class IdentifierInfo {
public:IdentifierInfo();
  const char *getNameStart() const {
    typedef std::pair < IdentifierInfo, const char *>actualtype;
    return ((const actualtype *) this)->second;
  }
  unsigned getLength() const {
    typedef std::pair < IdentifierInfo, const char *>actualtype;
    const char *p = ((const actualtype *) this)->second - 2;
    return (((unsigned) p[0]) | (((unsigned) p[1]) << 8)) - 1;
  }
  llvm::StringRef getName() const {
    return llvm::StringRef(getNameStart(), getLength());
  }
};
}
namespace llvm {
template < typename T, typename R = T > class StringSwitch {
  StringRef Str;
  const T *Result;
public:
  explicit StringSwitch(StringRef Str) : Str(Str), Result(0) {}
  template < unsigned N > StringSwitch & Case(const char (&S)[N],
                                              const T & Value) {
    return *this;
  }
  R Default(const T & Value) const {
    return Value;
  }
};
}

using namespace clang;

AttributeList::Kind AttributeList::getKind(const IdentifierInfo * Name) {
  llvm::StringRef AttrName = Name->getName();
  if (AttrName.startswith("__") && AttrName.endswith("__"))
    AttrName = AttrName.substr(2, AttrName.size() - 4);

  return llvm::StringSwitch < AttributeList::Kind > (AttrName)
    .Case("weak", AT_weak)
    .Case("weakref", AT_weakref)
    .Case("pure", AT_pure)
    .Case("mode", AT_mode)
    .Case("used", AT_used)
    .Case("alias", AT_alias)
    .Case("align", AT_aligned)
    .Case("final", AT_final)
    .Case("cdecl", AT_cdecl)
    .Case("const", AT_const)
    .Case("__const", AT_const)
    .Case("blocks", AT_blocks)
    .Case("format", AT_format)
    .Case("hiding", AT_hiding)
    .Case("malloc", AT_malloc)
    .Case("packed", AT_packed)
    .Case("unused", AT_unused)
    .Case("aligned", AT_aligned)
    .Case("cleanup", AT_cleanup)
    .Case("naked", AT_naked)
    .Case("nodebug", AT_nodebug)
    .Case("nonnull", AT_nonnull)
    .Case("nothrow", AT_nothrow)
    .Case("objc_gc", AT_objc_gc)
    .Case("regparm", AT_regparm)
    .Case("section", AT_section)
    .Case("stdcall", AT_stdcall)
    .Case("annotate", AT_annotate)
    .Case("fastcall", AT_fastcall)
    .Case("ibaction", AT_IBAction)
    .Case("iboutlet", AT_IBOutlet)
    .Case("iboutletcollection", AT_IBOutletCollection)
    .Case("noreturn", AT_noreturn)
    .Case("noinline", AT_noinline)
    .Case("override", AT_override)
    .Case("sentinel", AT_sentinel)
    .Case("NSObject", AT_nsobject)
    .Case("dllimport", AT_dllimport)
    .Case("dllexport", AT_dllexport)
    .Case("may_alias", IgnoredAttribute)
    .Case("base_check", AT_base_check)
    .Case("deprecated", AT_deprecated)
    .Case("visibility", AT_visibility)
    .Case("destructor", AT_destructor)
    .Case("format_arg", AT_format_arg)
    .Case("gnu_inline", AT_gnu_inline)
    .Case("weak_import", AT_weak_import)
    .Case("vecreturn", AT_vecreturn)
    .Case("vector_size", AT_vector_size)
    .Case("constructor", AT_constructor)
    .Case("unavailable", AT_unavailable)
    .Case("overloadable", AT_overloadable)
    .Case("address_space", AT_address_space)
    .Case("always_inline", AT_always_inline)
    .Case("returns_twice", AT_returns_twice)
    .Case("vec_type_hint", IgnoredAttribute)
    .Case("objc_exception", AT_objc_exception)
    .Case("ext_vector_type", AT_ext_vector_type)
    .Case("transparent_union", AT_transparent_union)
    .Case("analyzer_noreturn", AT_analyzer_noreturn)
    .Case("warn_unused_result", AT_warn_unused_result)
    .Case("carries_dependency", AT_carries_dependency)
    .Case("ns_returns_not_retained", AT_ns_returns_not_retained)
    .Case("ns_returns_retained", AT_ns_returns_retained)
    .Case("cf_returns_not_retained", AT_cf_returns_not_retained)
    .Case("cf_returns_retained", AT_cf_returns_retained)
    .Case("ownership_returns", AT_ownership_returns)
    .Case("ownership_holds", AT_ownership_holds)
    .Case("ownership_takes", AT_ownership_takes)
    .Case("reqd_work_group_size", AT_reqd_wg_size)
    .Case("init_priority", AT_init_priority)
    .Case("no_instrument_function", AT_no_instrument_function)
    .Case("thiscall", AT_thiscall)
    .Case("pascal", AT_pascal)
    .Case("__cdecl", AT_cdecl)
    .Case("__stdcall", AT_stdcall)
    .Case("__fastcall", AT_fastcall)
    .Case("__thiscall", AT_thiscall)
    .Case("__pascal", AT_pascal)
    .Default(UnknownAttribute);
}

// RUN: c-index-test -test-annotate-tokens=%s:1:1:186:1 -target x86_64-unknown-unknown %s 2>&1 | FileCheck -check-prefix=CHECK-tokens %s
// CHECK-tokens: Keyword: "typedef" [1:1 - 1:8]
// CHECK-tokens: Keyword: "long" [1:9 - 1:13]
// CHECK-tokens: Keyword: "unsigned" [1:14 - 1:22]
// CHECK-tokens: Keyword: "int" [1:23 - 1:26]
// CHECK-tokens: Identifier: "__darwin_size_t" [1:27 - 1:42] TypedefDecl=__darwin_size_t:1:27 (Definition)
// CHECK-tokens: Punctuation: ";" [1:42 - 1:43]
// CHECK-tokens: Keyword: "typedef" [2:1 - 2:8]
// CHECK-tokens: Identifier: "__darwin_size_t" [2:9 - 2:24]
// CHECK-tokens: Identifier: "size_t" [2:25 - 2:31] TypedefDecl=size_t:2:25 (Definition)
// CHECK-tokens: Punctuation: ";" [2:31 - 2:32]
// CHECK-tokens: Keyword: "namespace" [3:1 - 3:10]
// CHECK-tokens: Identifier: "std" [3:11 - 3:14] Namespace=std:3:11 (Definition)
// CHECK-tokens: Punctuation: "{" [3:15 - 3:16] Namespace=std:3:11 (Definition)
// CHECK-tokens: Keyword: "template" [4:3 - 4:11] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Punctuation: "<" [4:12 - 4:13] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Keyword: "class" [4:14 - 4:19] TemplateTypeParameter=_T1:4:20 (Definition)
// CHECK-tokens: Identifier: "_T1" [4:20 - 4:23] TemplateTypeParameter=_T1:4:20 (Definition)
// CHECK-tokens: Punctuation: "," [4:23 - 4:24] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Keyword: "class" [4:25 - 4:30] TemplateTypeParameter=_T2:4:31 (Definition)
// CHECK-tokens: Identifier: "_T2" [4:31 - 4:34] TemplateTypeParameter=_T2:4:31 (Definition)
// CHECK-tokens: Punctuation: ">" [4:35 - 4:36] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Keyword: "struct" [4:37 - 4:43] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Identifier: "pair" [4:44 - 4:48] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Punctuation: "{" [4:49 - 4:50] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Identifier: "_T2" [4:51 - 4:54] TypeRef=_T2:4:31
// CHECK-tokens: Identifier: "second" [4:55 - 4:61] FieldDecl=second:4:55 (Definition)
// CHECK-tokens: Punctuation: ";" [4:61 - 4:62] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Punctuation: "}" [4:63 - 4:64] ClassTemplate=pair:4:44 (Definition)
// CHECK-tokens: Punctuation: ";" [4:64 - 4:65] Namespace=std:3:11 (Definition)
// CHECK-tokens: Punctuation: "}" [5:1 - 5:2] Namespace=std:3:11 (Definition)
// CHECK-tokens: Keyword: "extern" [6:1 - 6:7]
// CHECK-tokens: Literal: ""C"" [6:8 - 6:11] UnexposedDecl=:6:8 (Definition)
// CHECK-tokens: Punctuation: "{" [6:12 - 6:13] UnexposedDecl=:6:8 (Definition)
// CHECK-tokens: Keyword: "int" [7:3 - 7:6] FunctionDecl=memcmp:7:7
// CHECK-tokens: Identifier: "memcmp" [7:7 - 7:13] FunctionDecl=memcmp:7:7
// CHECK-tokens: Punctuation: "(" [7:13 - 7:14] FunctionDecl=memcmp:7:7
// CHECK-tokens: Keyword: "const" [7:14 - 7:19] ParmDecl=:7:26 (Definition)
// CHECK-tokens: Keyword: "void" [7:20 - 7:24] ParmDecl=:7:26 (Definition)
// CHECK-tokens: Punctuation: "*" [7:25 - 7:26] ParmDecl=:7:26 (Definition)
// CHECK-tokens: Punctuation: "," [7:26 - 7:27] FunctionDecl=memcmp:7:7
// CHECK-tokens: Keyword: "const" [7:28 - 7:33] ParmDecl=:7:40 (Definition)
// CHECK-tokens: Keyword: "void" [7:34 - 7:38] ParmDecl=:7:40 (Definition)
// CHECK-tokens: Punctuation: "*" [7:39 - 7:40] ParmDecl=:7:40 (Definition)
// CHECK-tokens: Punctuation: "," [7:40 - 7:41] FunctionDecl=memcmp:7:7
// CHECK-tokens: Identifier: "size_t" [7:42 - 7:48] TypeRef=size_t:2:25
// CHECK-tokens: Punctuation: ")" [7:48 - 7:49] FunctionDecl=memcmp:7:7
// CHECK-tokens: Punctuation: ";" [7:49 - 7:50] UnexposedDecl=:6:8 (Definition)
// CHECK-tokens: Identifier: "size_t" [8:3 - 8:9] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "strlen" [8:10 - 8:16] FunctionDecl=strlen:8:10
// CHECK-tokens: Punctuation: "(" [8:16 - 8:17] FunctionDecl=strlen:8:10
// CHECK-tokens: Keyword: "const" [8:17 - 8:22] ParmDecl=:8:29 (Definition)
// CHECK-tokens: Keyword: "char" [8:23 - 8:27] ParmDecl=:8:29 (Definition)
// CHECK-tokens: Punctuation: "*" [8:28 - 8:29] ParmDecl=:8:29 (Definition)
// CHECK-tokens: Punctuation: ")" [8:29 - 8:30] FunctionDecl=strlen:8:10
// CHECK-tokens: Punctuation: ";" [8:30 - 8:31]
// CHECK-tokens: Punctuation: "}" [9:1 - 9:2]
// CHECK-tokens: Keyword: "namespace" [10:1 - 10:10]
// CHECK-tokens: Identifier: "clang" [10:17 - 10:22] Namespace=clang:10:17 (Definition)
// CHECK-tokens: Punctuation: "{" [10:23 - 10:24] Namespace=clang:10:17 (Definition)
// CHECK-tokens: Keyword: "class" [11:3 - 11:8] ClassDecl=IdentifierInfo:11:9
// CHECK-tokens: Identifier: "IdentifierInfo" [11:9 - 11:23] ClassDecl=IdentifierInfo:11:9
// CHECK-tokens: Punctuation: ";" [11:23 - 11:24] Namespace=clang:10:17 (Definition)
// CHECK-tokens: Keyword: "class" [12:3 - 12:8] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Identifier: "AttributeList" [12:9 - 12:22] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Punctuation: "{" [12:23 - 12:24] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Keyword: "enum" [13:5 - 13:9] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "Kind" [13:10 - 13:14] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Punctuation: "{" [13:15 - 13:16] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_IBAction" [14:7 - 14:18] EnumConstantDecl=AT_IBAction:14:7 (Definition)
// CHECK-tokens: Punctuation: "," [14:18 - 14:19] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_IBOutlet" [14:20 - 14:31] EnumConstantDecl=AT_IBOutlet:14:20 (Definition)
// CHECK-tokens: Punctuation: "," [14:31 - 14:32] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_IBOutletCollection" [14:33 - 14:54] EnumConstantDecl=AT_IBOutletCollection:14:33 (Definition)
// CHECK-tokens: Punctuation: "," [14:54 - 14:55] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_address_space" [15:7 - 15:23] EnumConstantDecl=AT_address_space:15:7 (Definition)
// CHECK-tokens: Punctuation: "," [15:23 - 15:24] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_alias" [15:25 - 15:33] EnumConstantDecl=AT_alias:15:25 (Definition)
// CHECK-tokens: Punctuation: "," [15:33 - 15:34] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_aligned" [15:35 - 15:45] EnumConstantDecl=AT_aligned:15:35 (Definition)
// CHECK-tokens: Punctuation: "," [15:45 - 15:46] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_always_inline" [15:47 - 15:63] EnumConstantDecl=AT_always_inline:15:47 (Definition)
// CHECK-tokens: Punctuation: "," [15:63 - 15:64] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_analyzer_noreturn" [16:7 - 16:27] EnumConstantDecl=AT_analyzer_noreturn:16:7 (Definition)
// CHECK-tokens: Punctuation: "," [16:27 - 16:28] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_annotate" [16:29 - 16:40] EnumConstantDecl=AT_annotate:16:29 (Definition)
// CHECK-tokens: Punctuation: "," [16:40 - 16:41] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_base_check" [16:42 - 16:55] EnumConstantDecl=AT_base_check:16:42 (Definition)
// CHECK-tokens: Punctuation: "," [16:55 - 16:56] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_blocks" [16:57 - 16:66] EnumConstantDecl=AT_blocks:16:57 (Definition)
// CHECK-tokens: Punctuation: "," [16:66 - 16:67] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_carries_dependency" [17:7 - 17:28] EnumConstantDecl=AT_carries_dependency:17:7 (Definition)
// CHECK-tokens: Punctuation: "," [17:28 - 17:29] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_cdecl" [17:30 - 17:38] EnumConstantDecl=AT_cdecl:17:30 (Definition)
// CHECK-tokens: Punctuation: "," [17:38 - 17:39] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_cleanup" [17:40 - 17:50] EnumConstantDecl=AT_cleanup:17:40 (Definition)
// CHECK-tokens: Punctuation: "," [17:50 - 17:51] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_const" [17:52 - 17:60] EnumConstantDecl=AT_const:17:52 (Definition)
// CHECK-tokens: Punctuation: "," [17:60 - 17:61] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_constructor" [17:62 - 17:76] EnumConstantDecl=AT_constructor:17:62 (Definition)
// CHECK-tokens: Punctuation: "," [17:76 - 17:77] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_deprecated" [18:7 - 18:20] EnumConstantDecl=AT_deprecated:18:7 (Definition)
// CHECK-tokens: Punctuation: "," [18:20 - 18:21] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_destructor" [18:22 - 18:35] EnumConstantDecl=AT_destructor:18:22 (Definition)
// CHECK-tokens: Punctuation: "," [18:35 - 18:36] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_dllexport" [18:37 - 18:49] EnumConstantDecl=AT_dllexport:18:37 (Definition)
// CHECK-tokens: Punctuation: "," [18:49 - 18:50] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_dllimport" [18:51 - 18:63] EnumConstantDecl=AT_dllimport:18:51 (Definition)
// CHECK-tokens: Punctuation: "," [18:63 - 18:64] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ext_vector_type" [19:7 - 19:25] EnumConstantDecl=AT_ext_vector_type:19:7 (Definition)
// CHECK-tokens: Punctuation: "," [19:25 - 19:26] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_fastcall" [19:27 - 19:38] EnumConstantDecl=AT_fastcall:19:27 (Definition)
// CHECK-tokens: Punctuation: "," [19:38 - 19:39] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_final" [19:40 - 19:48] EnumConstantDecl=AT_final:19:40 (Definition)
// CHECK-tokens: Punctuation: "," [19:48 - 19:49] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_format" [19:50 - 19:59] EnumConstantDecl=AT_format:19:50 (Definition)
// CHECK-tokens: Punctuation: "," [19:59 - 19:60] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_format_arg" [19:61 - 19:74] EnumConstantDecl=AT_format_arg:19:61 (Definition)
// CHECK-tokens: Punctuation: "," [19:74 - 19:75] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_gnu_inline" [20:7 - 20:20] EnumConstantDecl=AT_gnu_inline:20:7 (Definition)
// CHECK-tokens: Punctuation: "," [20:20 - 20:21] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_hiding" [20:22 - 20:31] EnumConstantDecl=AT_hiding:20:22 (Definition)
// CHECK-tokens: Punctuation: "," [20:31 - 20:32] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_malloc" [20:33 - 20:42] EnumConstantDecl=AT_malloc:20:33 (Definition)
// CHECK-tokens: Punctuation: "," [20:42 - 20:43] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_mode" [20:44 - 20:51] EnumConstantDecl=AT_mode:20:44 (Definition)
// CHECK-tokens: Punctuation: "," [20:51 - 20:52] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_naked" [20:53 - 20:61] EnumConstantDecl=AT_naked:20:53 (Definition)
// CHECK-tokens: Punctuation: "," [20:61 - 20:62] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_nodebug" [20:63 - 20:73] EnumConstantDecl=AT_nodebug:20:63 (Definition)
// CHECK-tokens: Punctuation: "," [20:73 - 20:74] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_noinline" [21:7 - 21:18] EnumConstantDecl=AT_noinline:21:7 (Definition)
// CHECK-tokens: Punctuation: "," [21:18 - 21:19] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_no_instrument_function" [21:20 - 21:45] EnumConstantDecl=AT_no_instrument_function:21:20 (Definition)
// CHECK-tokens: Punctuation: "," [21:45 - 21:46] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_nonnull" [21:47 - 21:57] EnumConstantDecl=AT_nonnull:21:47 (Definition)
// CHECK-tokens: Punctuation: "," [21:57 - 21:58] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_noreturn" [21:59 - 21:70] EnumConstantDecl=AT_noreturn:21:59 (Definition)
// CHECK-tokens: Punctuation: "," [21:70 - 21:71] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_nothrow" [22:7 - 22:17] EnumConstantDecl=AT_nothrow:22:7 (Definition)
// CHECK-tokens: Punctuation: "," [22:17 - 22:18] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_nsobject" [22:19 - 22:30] EnumConstantDecl=AT_nsobject:22:19 (Definition)
// CHECK-tokens: Punctuation: "," [22:30 - 22:31] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_objc_exception" [22:32 - 22:49] EnumConstantDecl=AT_objc_exception:22:32 (Definition)
// CHECK-tokens: Punctuation: "," [22:49 - 22:50] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_override" [22:51 - 22:62] EnumConstantDecl=AT_override:22:51 (Definition)
// CHECK-tokens: Punctuation: "," [22:62 - 22:63] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_cf_returns_not_retained" [23:7 - 23:33] EnumConstantDecl=AT_cf_returns_not_retained:23:7 (Definition)
// CHECK-tokens: Punctuation: "," [23:33 - 23:34] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_cf_returns_retained" [23:35 - 23:57] EnumConstantDecl=AT_cf_returns_retained:23:35 (Definition)
// CHECK-tokens: Punctuation: "," [23:57 - 23:58] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ns_returns_not_retained" [24:7 - 24:33] EnumConstantDecl=AT_ns_returns_not_retained:24:7 (Definition)
// CHECK-tokens: Punctuation: "," [24:33 - 24:34] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ns_returns_retained" [24:35 - 24:57] EnumConstantDecl=AT_ns_returns_retained:24:35 (Definition)
// CHECK-tokens: Punctuation: "," [24:57 - 24:58] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_objc_gc" [24:59 - 24:69] EnumConstantDecl=AT_objc_gc:24:59 (Definition)
// CHECK-tokens: Punctuation: "," [24:69 - 24:70] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_overloadable" [25:7 - 25:22] EnumConstantDecl=AT_overloadable:25:7 (Definition)
// CHECK-tokens: Punctuation: "," [25:22 - 25:23] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ownership_holds" [25:24 - 25:42] EnumConstantDecl=AT_ownership_holds:25:24 (Definition)
// CHECK-tokens: Punctuation: "," [25:42 - 25:43] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ownership_returns" [25:44 - 25:64] EnumConstantDecl=AT_ownership_returns:25:44 (Definition)
// CHECK-tokens: Punctuation: "," [25:64 - 25:65] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_ownership_takes" [26:7 - 26:25] EnumConstantDecl=AT_ownership_takes:26:7 (Definition)
// CHECK-tokens: Punctuation: "," [26:25 - 26:26] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_packed" [26:27 - 26:36] EnumConstantDecl=AT_packed:26:27 (Definition)
// CHECK-tokens: Punctuation: "," [26:36 - 26:37] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_pascal" [26:38 - 26:47] EnumConstantDecl=AT_pascal:26:38 (Definition)
// CHECK-tokens: Punctuation: "," [26:47 - 26:48] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_pure" [26:49 - 26:56] EnumConstantDecl=AT_pure:26:49 (Definition)
// CHECK-tokens: Punctuation: "," [26:56 - 26:57] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_regparm" [26:58 - 26:68] EnumConstantDecl=AT_regparm:26:58 (Definition)
// CHECK-tokens: Punctuation: "," [26:68 - 26:69] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_section" [27:7 - 27:17] EnumConstantDecl=AT_section:27:7 (Definition)
// CHECK-tokens: Punctuation: "," [27:17 - 27:18] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_sentinel" [27:19 - 27:30] EnumConstantDecl=AT_sentinel:27:19 (Definition)
// CHECK-tokens: Punctuation: "," [27:30 - 27:31] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_stdcall" [27:32 - 27:42] EnumConstantDecl=AT_stdcall:27:32 (Definition)
// CHECK-tokens: Punctuation: "," [27:42 - 27:43] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_thiscall" [27:44 - 27:55] EnumConstantDecl=AT_thiscall:27:44 (Definition)
// CHECK-tokens: Punctuation: "," [27:55 - 27:56] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_transparent_union" [27:57 - 27:77] EnumConstantDecl=AT_transparent_union:27:57 (Definition)
// CHECK-tokens: Punctuation: "," [27:77 - 27:78] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_unavailable" [28:7 - 28:21] EnumConstantDecl=AT_unavailable:28:7 (Definition)
// CHECK-tokens: Punctuation: "," [28:21 - 28:22] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_unused" [28:23 - 28:32] EnumConstantDecl=AT_unused:28:23 (Definition)
// CHECK-tokens: Punctuation: "," [28:32 - 28:33] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_used" [28:34 - 28:41] EnumConstantDecl=AT_used:28:34 (Definition)
// CHECK-tokens: Punctuation: "," [28:41 - 28:42] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_vecreturn" [28:43 - 28:55] EnumConstantDecl=AT_vecreturn:28:43 (Definition)
// CHECK-tokens: Punctuation: "," [28:55 - 28:56] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_vector_size" [28:57 - 28:71] EnumConstantDecl=AT_vector_size:28:57 (Definition)
// CHECK-tokens: Punctuation: "," [28:71 - 28:72] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_visibility" [29:7 - 29:20] EnumConstantDecl=AT_visibility:29:7 (Definition)
// CHECK-tokens: Punctuation: "," [29:20 - 29:21] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_warn_unused_result" [29:22 - 29:43] EnumConstantDecl=AT_warn_unused_result:29:22 (Definition)
// CHECK-tokens: Punctuation: "," [29:43 - 29:44] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_weak" [29:45 - 29:52] EnumConstantDecl=AT_weak:29:45 (Definition)
// CHECK-tokens: Punctuation: "," [29:52 - 29:53] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_weakref" [29:54 - 29:64] EnumConstantDecl=AT_weakref:29:54 (Definition)
// CHECK-tokens: Punctuation: "," [29:64 - 29:65] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_weak_import" [30:7 - 30:21] EnumConstantDecl=AT_weak_import:30:7 (Definition)
// CHECK-tokens: Punctuation: "," [30:21 - 30:22] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_reqd_wg_size" [30:23 - 30:38] EnumConstantDecl=AT_reqd_wg_size:30:23 (Definition)
// CHECK-tokens: Punctuation: "," [30:38 - 30:39] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_init_priority" [30:40 - 30:56] EnumConstantDecl=AT_init_priority:30:40 (Definition)
// CHECK-tokens: Punctuation: "," [30:56 - 30:57] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "AT_returns_twice" [31:7 - 31:23] EnumConstantDecl=AT_returns_twice:31:7 (Definition)
// CHECK-tokens: Punctuation: "," [31:23 - 31:24] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Identifier: "IgnoredAttribute" [31:25 - 31:41] EnumConstantDecl=IgnoredAttribute:31:25 (Definition)
// CHECK-tokens: Punctuation: "}" [32:5 - 32:6] EnumDecl=Kind:13:10 (Definition)
// CHECK-tokens: Punctuation: ";" [32:6 - 32:7] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Keyword: "static" [33:5 - 33:11] CXXMethod=getKind:33:17 (static)
// CHECK-tokens: Identifier: "Kind" [33:12 - 33:16] TypeRef=enum clang::AttributeList::Kind:13:10
// CHECK-tokens: Identifier: "getKind" [33:17 - 33:24] CXXMethod=getKind:33:17 (static)
// CHECK-tokens: Punctuation: "(" [33:24 - 33:25] CXXMethod=getKind:33:17 (static)
// CHECK-tokens: Keyword: "const" [33:25 - 33:30] ParmDecl=Name:33:48 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [33:31 - 33:45] TypeRef=class clang::IdentifierInfo:66:7
// CHECK-tokens: Punctuation: "*" [33:46 - 33:47] ParmDecl=Name:33:48 (Definition)
// CHECK-tokens: Identifier: "Name" [33:48 - 33:52] ParmDecl=Name:33:48 (Definition)
// CHECK-tokens: Punctuation: ")" [33:52 - 33:53] CXXMethod=getKind:33:17 (static)
// CHECK-tokens: Punctuation: ";" [33:53 - 33:54] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Punctuation: "}" [34:3 - 34:4] ClassDecl=AttributeList:12:9 (Definition)
// CHECK-tokens: Punctuation: ";" [34:4 - 34:5] Namespace=clang:10:17 (Definition)
// CHECK-tokens: Punctuation: "}" [35:1 - 35:2] Namespace=clang:10:17 (Definition)
// CHECK-tokens: Identifier: "size_t" [36:1 - 36:7] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "magic_length" [36:8 - 36:20] FunctionDecl=magic_length:36:8
// CHECK-tokens: Punctuation: "(" [36:20 - 36:21] FunctionDecl=magic_length:36:8
// CHECK-tokens: Keyword: "const" [36:21 - 36:26] ParmDecl=s:36:33 (Definition)
// CHECK-tokens: Keyword: "char" [36:27 - 36:31] ParmDecl=s:36:33 (Definition)
// CHECK-tokens: Punctuation: "*" [36:32 - 36:33] ParmDecl=s:36:33 (Definition)
// CHECK-tokens: Identifier: "s" [36:33 - 36:34] ParmDecl=s:36:33 (Definition)
// CHECK-tokens: Punctuation: ")" [36:34 - 36:35] FunctionDecl=magic_length:36:8
// CHECK-tokens: Punctuation: ";" [36:35 - 36:36]
// CHECK-tokens: Keyword: "namespace" [37:1 - 37:10]
// CHECK-tokens: Identifier: "llvm" [37:11 - 37:15] Namespace=llvm:37:11 (Definition)
// CHECK-tokens: Punctuation: "{" [37:16 - 37:17] Namespace=llvm:37:11 (Definition)
// CHECK-tokens: Keyword: "class" [38:1 - 38:6] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Identifier: "StringRef" [38:7 - 38:16] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Punctuation: "{" [38:17 - 38:18] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Keyword: "public" [39:1 - 39:7] CXXAccessSpecifier=:39:1 (Definition)
// CHECK-tokens: Punctuation: ":" [39:7 - 39:8] CXXAccessSpecifier=:39:1 (Definition)
// CHECK-tokens: Keyword: "typedef" [40:3 - 40:10] TypedefDecl=iterator:40:23 (Definition)
// CHECK-tokens: Keyword: "const" [40:11 - 40:16] TypedefDecl=iterator:40:23 (Definition)
// CHECK-tokens: Keyword: "char" [40:17 - 40:21] TypedefDecl=iterator:40:23 (Definition)
// CHECK-tokens: Punctuation: "*" [40:22 - 40:23] TypedefDecl=iterator:40:23 (Definition)
// CHECK-tokens: Identifier: "iterator" [40:23 - 40:31] TypedefDecl=iterator:40:23 (Definition)
// CHECK-tokens: Punctuation: ";" [40:31 - 40:32] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Keyword: "static" [41:3 - 41:9] VarDecl=npos:41:23
// CHECK-tokens: Keyword: "const" [41:10 - 41:15] VarDecl=npos:41:23
// CHECK-tokens: Identifier: "size_t" [41:16 - 41:22] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "npos" [41:23 - 41:27] VarDecl=npos:41:23
// CHECK-tokens: Punctuation: "=" [41:28 - 41:29] VarDecl=npos:41:23
// CHECK-tokens: Punctuation: "~" [41:30 - 41:31] UnaryOperator=
// CHECK-tokens: Identifier: "size_t" [41:31 - 41:37] TypeRef=size_t:2:25
// CHECK-tokens: Punctuation: "(" [41:37 - 41:38] CXXFunctionalCastExpr=
// CHECK-tokens: Literal: "0" [41:38 - 41:39] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [41:39 - 41:40] CXXFunctionalCastExpr
// CHECK-tokens: Punctuation: ";" [41:40 - 41:41] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Keyword: "private" [42:1 - 42:8] CXXAccessSpecifier=:42:1 (Definition)
// CHECK-tokens: Punctuation: ":" [42:8 - 42:9] CXXAccessSpecifier=:42:1 (Definition)
// CHECK-tokens: Keyword: "const" [43:3 - 43:8] FieldDecl=Data:43:15 (Definition)
// CHECK-tokens: Keyword: "char" [43:9 - 43:13] FieldDecl=Data:43:15 (Definition)
// CHECK-tokens: Punctuation: "*" [43:14 - 43:15] FieldDecl=Data:43:15 (Definition)
// CHECK-tokens: Identifier: "Data" [43:15 - 43:19] FieldDecl=Data:43:15 (Definition)
// CHECK-tokens: Punctuation: ";" [43:19 - 43:20] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Identifier: "size_t" [44:3 - 44:9] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "Length" [44:10 - 44:16] FieldDecl=Length:44:10 (Definition)
// CHECK-tokens: Punctuation: ";" [44:16 - 44:17] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Keyword: "static" [45:3 - 45:9] CXXMethod=min:45:17 (Definition) (static)
// CHECK-tokens: Identifier: "size_t" [45:10 - 45:16] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "min" [45:17 - 45:20] CXXMethod=min:45:17 (Definition) (static)
// CHECK-tokens: Punctuation: "(" [45:20 - 45:21] CXXMethod=min:45:17 (Definition) (static)
// CHECK-tokens: Identifier: "size_t" [45:21 - 45:27] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "a" [45:28 - 45:29] ParmDecl=a:45:28 (Definition)
// CHECK-tokens: Punctuation: "," [45:29 - 45:30] CXXMethod=min:45:17 (Definition) (static)
// CHECK-tokens: Identifier: "size_t" [45:31 - 45:37] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "b" [45:38 - 45:39] ParmDecl=b:45:38 (Definition)
// CHECK-tokens: Punctuation: ")" [45:39 - 45:40] CXXMethod=min:45:17 (Definition) (static)
// CHECK-tokens: Punctuation: "{" [45:41 - 45:42] CompoundStmt=
// CHECK-tokens: Keyword: "return" [45:43 - 45:49] ReturnStmt=
// CHECK-tokens: Identifier: "a" [45:50 - 45:51] DeclRefExpr=a:45:28
// CHECK-tokens: Punctuation: "<" [45:52 - 45:53] BinaryOperator=
// CHECK-tokens: Identifier: "b" [45:54 - 45:55] DeclRefExpr=b:45:38
// CHECK-tokens: Punctuation: "?" [45:56 - 45:57] ConditionalOperator=
// CHECK-tokens: Identifier: "a" [45:58 - 45:59] DeclRefExpr=a:45:28
// CHECK-tokens: Punctuation: ":" [45:60 - 45:61] ConditionalOperator
// CHECK-tokens: Identifier: "b" [45:62 - 45:63] DeclRefExpr=b:45:38
// CHECK-tokens: Punctuation: ";" [45:63 - 45:64] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [45:65 - 45:66] CompoundStmt=
// CHECK-tokens: Keyword: "public" [46:1 - 46:7] CXXAccessSpecifier=:46:1 (Definition)
// CHECK-tokens: Punctuation: ":" [46:7 - 46:8] CXXAccessSpecifier=:46:1 (Definition)
// CHECK-tokens: Identifier: "StringRef" [47:3 - 47:12] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Punctuation: "(" [47:12 - 47:13] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Punctuation: ")" [47:13 - 47:14] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Punctuation: ":" [47:14 - 47:15] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Identifier: "Data" [47:16 - 47:20] MemberRef=Data:43:15
// CHECK-tokens: Punctuation: "(" [47:20 - 47:21] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Literal: "0" [47:21 - 47:22] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [47:22 - 47:23] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Punctuation: "," [47:23 - 47:24] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Identifier: "Length" [47:25 - 47:31] MemberRef=Length:44:10
// CHECK-tokens: Punctuation: "(" [47:31 - 47:32] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Literal: "0" [47:32 - 47:33] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [47:33 - 47:34] CXXConstructor=StringRef:47:3 (Definition)
// CHECK-tokens: Punctuation: "{" [47:35 - 47:36] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [47:36 - 47:37] CompoundStmt=
// CHECK-tokens: Identifier: "StringRef" [48:3 - 48:12] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Punctuation: "(" [48:12 - 48:13] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Keyword: "const" [48:13 - 48:18] ParmDecl=Str:48:25 (Definition)
// CHECK-tokens: Keyword: "char" [48:19 - 48:23] ParmDecl=Str:48:25 (Definition)
// CHECK-tokens: Punctuation: "*" [48:24 - 48:25] ParmDecl=Str:48:25 (Definition)
// CHECK-tokens: Identifier: "Str" [48:25 - 48:28] ParmDecl=Str:48:25 (Definition)
// CHECK-tokens: Punctuation: ")" [48:28 - 48:29] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Punctuation: ":" [48:30 - 48:31] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Identifier: "Data" [48:32 - 48:36] MemberRef=Data:43:15
// CHECK-tokens: Punctuation: "(" [48:36 - 48:37] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Identifier: "Str" [48:37 - 48:40] DeclRefExpr=Str:48:25
// CHECK-tokens: Punctuation: ")" [48:40 - 48:41] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Punctuation: "," [48:41 - 48:42] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Identifier: "Length" [48:43 - 48:49] MemberRef=Length:44:10
// CHECK-tokens: Punctuation: "(" [48:49 - 48:50] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Identifier: "magic_length" [48:50 - 48:62] DeclRefExpr=magic_length:36:8
// CHECK-tokens: Punctuation: "(" [48:62 - 48:63] CallExpr=magic_length:36:8
// CHECK-tokens: Identifier: "Str" [48:63 - 48:66] DeclRefExpr=Str:48:25
// CHECK-tokens: Punctuation: ")" [48:66 - 48:67] CallExpr=magic_length:36:8
// CHECK-tokens: Punctuation: ")" [48:67 - 48:68] CXXConstructor=StringRef:48:3 (Definition)
// CHECK-tokens: Punctuation: "{" [48:69 - 48:70] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [48:70 - 48:71] CompoundStmt=
// CHECK-tokens: Identifier: "StringRef" [49:3 - 49:12] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Punctuation: "(" [49:12 - 49:13] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Keyword: "const" [49:13 - 49:18] ParmDecl=data:49:25 (Definition)
// CHECK-tokens: Keyword: "char" [49:19 - 49:23] ParmDecl=data:49:25 (Definition)
// CHECK-tokens: Punctuation: "*" [49:24 - 49:25] ParmDecl=data:49:25 (Definition)
// CHECK-tokens: Identifier: "data" [49:25 - 49:29] ParmDecl=data:49:25 (Definition)
// CHECK-tokens: Punctuation: "," [49:29 - 49:30] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Identifier: "size_t" [49:31 - 49:37] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "length" [49:38 - 49:44] ParmDecl=length:49:38 (Definition)
// CHECK-tokens: Punctuation: ")" [49:44 - 49:45] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Punctuation: ":" [49:46 - 49:47] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Identifier: "Data" [49:48 - 49:52] MemberRef=Data:43:15
// CHECK-tokens: Punctuation: "(" [49:52 - 49:53] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Identifier: "data" [49:53 - 49:57] DeclRefExpr=data:49:25
// CHECK-tokens: Punctuation: ")" [49:57 - 49:58] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Punctuation: "," [49:58 - 49:59] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Identifier: "Length" [49:60 - 49:66] MemberRef=Length:44:10
// CHECK-tokens: Punctuation: "(" [49:66 - 49:67] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Identifier: "length" [49:67 - 49:73] DeclRefExpr=length:49:38
// CHECK-tokens: Punctuation: ")" [49:73 - 49:74] CXXConstructor=StringRef:49:3 (Definition)
// CHECK-tokens: Punctuation: "{" [49:75 - 49:76] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [49:76 - 49:77] CompoundStmt=
// CHECK-tokens: Identifier: "iterator" [50:3 - 50:11] TypeRef=iterator:40:23
// CHECK-tokens: Identifier: "end" [50:12 - 50:15] CXXMethod=end:50:12 (Definition)
// CHECK-tokens: Punctuation: "(" [50:15 - 50:16] CXXMethod=end:50:12 (Definition)
// CHECK-tokens: Punctuation: ")" [50:16 - 50:17] CXXMethod=end:50:12 (Definition)
// CHECK-tokens: Keyword: "const" [50:18 - 50:23] CXXMethod=end:50:12 (Definition)
// CHECK-tokens: Punctuation: "{" [50:24 - 50:25] CompoundStmt=
// CHECK-tokens: Keyword: "return" [50:26 - 50:32] ReturnStmt=
// CHECK-tokens: Identifier: "Data" [50:33 - 50:37]  MemberRefExpr=Data:43:15
// CHECK-tokens: Punctuation: ";" [50:37 - 50:38] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [50:39 - 50:40] CompoundStmt=
// CHECK-tokens: Identifier: "size_t" [51:3 - 51:9] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "size" [51:10 - 51:14] CXXMethod=size:51:10 (Definition)
// CHECK-tokens: Punctuation: "(" [51:14 - 51:15] CXXMethod=size:51:10 (Definition)
// CHECK-tokens: Punctuation: ")" [51:15 - 51:16] CXXMethod=size:51:10 (Definition)
// CHECK-tokens: Keyword: "const" [51:17 - 51:22] CXXMethod=size:51:10 (Definition)
// CHECK-tokens: Punctuation: "{" [51:23 - 51:24] CompoundStmt=
// CHECK-tokens: Keyword: "return" [51:25 - 51:31] ReturnStmt=
// CHECK-tokens: Identifier: "Length" [51:32 - 51:38] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: ";" [51:38 - 51:39] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [51:40 - 51:41] CompoundStmt=
// CHECK-tokens: Keyword: "bool" [52:3 - 52:7] CXXMethod=startswith:52:8 (Definition)
// CHECK-tokens: Identifier: "startswith" [52:8 - 52:18] CXXMethod=startswith:52:8 (Definition)
// CHECK-tokens: Punctuation: "(" [52:18 - 52:19] CXXMethod=startswith:52:8 (Definition)
// CHECK-tokens: Identifier: "StringRef" [52:19 - 52:28] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "Prefix" [52:29 - 52:35] ParmDecl=Prefix:52:29 (Definition)
// CHECK-tokens: Punctuation: ")" [52:35 - 52:36] CXXMethod=startswith:52:8 (Definition)
// CHECK-tokens: Keyword: "const" [52:37 - 52:42] CXXMethod=startswith:52:8 (Definition)
// CHECK-tokens: Punctuation: "{" [52:43 - 52:44] CompoundStmt=
// CHECK-tokens: Keyword: "return" [53:5 - 53:11] ReturnStmt=
// CHECK-tokens: Identifier: "Length" [53:12 - 53:18] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: ">=" [53:19 - 53:21] BinaryOperator=
// CHECK-tokens: Identifier: "Prefix" [53:22 - 53:28] DeclRefExpr=Prefix:52:29
// CHECK-tokens: Punctuation: "." [53:28 - 53:29] MemberRefExpr=Length:44:10
// CHECK-tokens: Identifier: "Length" [53:29 - 53:35] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: "&&" [53:36 - 53:38] BinaryOperator=
// CHECK-tokens: Identifier: "memcmp" [54:11 - 54:17] DeclRefExpr=memcmp:7:7
// CHECK-tokens: Punctuation: "(" [54:17 - 54:18] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "Data" [54:18 - 54:22]  MemberRefExpr=Data:43:15
// CHECK-tokens: Punctuation: "," [54:22 - 54:23] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "Prefix" [54:24 - 54:30] DeclRefExpr=Prefix:52:29
// CHECK-tokens: Punctuation: "." [54:30 - 54:31] MemberRefExpr=Data:43:15
// CHECK-tokens: Identifier: "Data" [54:31 - 54:35] MemberRefExpr=Data:43:15
// CHECK-tokens: Punctuation: "," [54:35 - 54:36] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "Prefix" [54:37 - 54:43] DeclRefExpr=Prefix:52:29
// CHECK-tokens: Punctuation: "." [54:43 - 54:44] MemberRefExpr=Length:44:10
// CHECK-tokens: Identifier: "Length" [54:44 - 54:50] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: ")" [54:50 - 54:51] CallExpr=memcmp:7:7
// CHECK-tokens: Punctuation: "==" [54:52 - 54:54] BinaryOperator=
// CHECK-tokens: Literal: "0" [54:55 - 54:56] IntegerLiteral=
// CHECK-tokens: Punctuation: ";" [54:56 - 54:57] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [55:3 - 55:4] CompoundStmt=
// CHECK-tokens: Keyword: "bool" [56:3 - 56:7] CXXMethod=endswith:56:8 (Definition)
// CHECK-tokens: Identifier: "endswith" [56:8 - 56:16] CXXMethod=endswith:56:8 (Definition)
// CHECK-tokens: Punctuation: "(" [56:16 - 56:17] CXXMethod=endswith:56:8 (Definition)
// CHECK-tokens: Identifier: "StringRef" [56:17 - 56:26] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "Suffix" [56:27 - 56:33] ParmDecl=Suffix:56:27 (Definition)
// CHECK-tokens: Punctuation: ")" [56:33 - 56:34] CXXMethod=endswith:56:8 (Definition)
// CHECK-tokens: Keyword: "const" [56:35 - 56:40] CXXMethod=endswith:56:8 (Definition)
// CHECK-tokens: Punctuation: "{" [56:41 - 56:42] CompoundStmt=
// CHECK-tokens: Keyword: "return" [57:5 - 57:11] ReturnStmt=
// CHECK-tokens: Identifier: "Length" [57:12 - 57:18] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: ">=" [57:19 - 57:21] BinaryOperator=
// CHECK-tokens: Identifier: "Suffix" [57:22 - 57:28] DeclRefExpr=Suffix:56:27
// CHECK-tokens: Punctuation: "." [57:28 - 57:29] MemberRefExpr=Length:44:10
// CHECK-tokens: Identifier: "Length" [57:29 - 57:35] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: "&&" [57:36 - 57:38] BinaryOperator=
// CHECK-tokens: Identifier: "memcmp" [58:7 - 58:13] DeclRefExpr=memcmp:7:7
// CHECK-tokens: Punctuation: "(" [58:13 - 58:14] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "end" [58:14 - 58:17] MemberRefExpr=end:50:12
// CHECK-tokens: Punctuation: "(" [58:17 - 58:18] CallExpr=end:50:12
// CHECK-tokens: Punctuation: ")" [58:18 - 58:19] CallExpr=end:50:12
// CHECK-tokens: Punctuation: "-" [58:20 - 58:21] BinaryOperator=
// CHECK-tokens: Identifier: "Suffix" [58:22 - 58:28] DeclRefExpr=Suffix:56:27
// CHECK-tokens: Punctuation: "." [58:28 - 58:29] MemberRefExpr=Length:44:10
// CHECK-tokens: Identifier: "Length" [58:29 - 58:35] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: "," [58:35 - 58:36] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "Suffix" [58:37 - 58:43] DeclRefExpr=Suffix:56:27
// CHECK-tokens: Punctuation: "." [58:43 - 58:44] MemberRefExpr=Data:43:15
// CHECK-tokens: Identifier: "Data" [58:44 - 58:48] MemberRefExpr=Data:43:15
// CHECK-tokens: Punctuation: "," [58:48 - 58:49] CallExpr=memcmp:7:7
// CHECK-tokens: Identifier: "Suffix" [58:50 - 58:56] DeclRefExpr=Suffix:56:27
// CHECK-tokens: Punctuation: "." [58:56 - 58:57] MemberRefExpr=Length:44:10
// CHECK-tokens: Identifier: "Length" [58:57 - 58:63] MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: ")" [58:63 - 58:64] CallExpr=memcmp:7:7
// CHECK-tokens: Punctuation: "==" [58:65 - 58:67] BinaryOperator=
// CHECK-tokens: Literal: "0" [58:68 - 58:69] IntegerLiteral=
// CHECK-tokens: Punctuation: ";" [58:69 - 58:70] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [59:3 - 59:4] CompoundStmt=
// CHECK-tokens: Identifier: "StringRef" [60:3 - 60:12] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "substr" [60:13 - 60:19] CXXMethod=substr:60:13 (Definition)
// CHECK-tokens: Punctuation: "(" [60:19 - 60:20] CXXMethod=substr:60:13 (Definition)
// CHECK-tokens: Identifier: "size_t" [60:20 - 60:26] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "Start" [60:27 - 60:32] ParmDecl=Start:60:27 (Definition)
// CHECK-tokens: Punctuation: "," [60:32 - 60:33] CXXMethod=substr:60:13 (Definition)
// CHECK-tokens: Identifier: "size_t" [60:34 - 60:40] TypeRef=size_t:2:25
// CHECK-tokens: Identifier: "N" [60:41 - 60:42] ParmDecl=N:60:41 (Definition)
// CHECK-tokens: Punctuation: "=" [60:43 - 60:44] ParmDecl=N:60:41 (Definition)
// CHECK-tokens: Identifier: "npos" [60:45 - 60:49] DeclRefExpr=npos:41:23
// CHECK-tokens: Punctuation: ")" [60:49 - 60:50] CXXMethod=substr:60:13 (Definition)
// CHECK-tokens: Keyword: "const" [60:51 - 60:56] CXXMethod=substr:60:13 (Definition)
// CHECK-tokens: Punctuation: "{" [60:57 - 60:58] CompoundStmt=
// CHECK-tokens: Keyword: "return" [61:5 - 61:11] ReturnStmt=
// CHECK-tokens: Identifier: "StringRef" [61:12 - 61:21] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Punctuation: "(" [61:21 - 61:22] CallExpr=StringRef:49:3
// CHECK-tokens: Identifier: "Data" [61:22 - 61:26]  MemberRefExpr=Data:43:15
// CHECK-tokens: Punctuation: "+" [61:27 - 61:28] BinaryOperator=
// CHECK-tokens: Identifier: "Start" [61:29 - 61:34] DeclRefExpr=Start:60:27
// CHECK-tokens: Punctuation: "," [61:34 - 61:35] CallExpr=StringRef:49:3
// CHECK-tokens: Identifier: "min" [61:36 - 61:39] DeclRefExpr=min:45:17
// CHECK-tokens: Punctuation: "(" [61:39 - 61:40] CallExpr=min:45:17
// CHECK-tokens: Identifier: "N" [61:40 - 61:41] DeclRefExpr=N:60:41
// CHECK-tokens: Punctuation: "," [61:41 - 61:42] CallExpr=min:45:17
// CHECK-tokens: Identifier: "Length" [61:43 - 61:49]  MemberRefExpr=Length:44:10
// CHECK-tokens: Punctuation: "-" [61:50 - 61:51] BinaryOperator=
// CHECK-tokens: Identifier: "Start" [61:52 - 61:57] DeclRefExpr=Start:60:27
// CHECK-tokens: Punctuation: ")" [61:57 - 61:58] CallExpr=min:45:17
// CHECK-tokens: Punctuation: ")" [61:58 - 61:59] CallExpr=StringRef:49:3
// CHECK-tokens: Punctuation: ";" [61:59 - 61:60] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [62:3 - 62:4] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [63:1 - 63:2] ClassDecl=StringRef:38:7 (Definition)
// CHECK-tokens: Punctuation: ";" [63:2 - 63:3] Namespace=llvm:37:11 (Definition)
// CHECK-tokens: Punctuation: "}" [64:1 - 64:2] Namespace=llvm:37:11 (Definition)
// CHECK-tokens: Keyword: "namespace" [65:1 - 65:10]
// CHECK-tokens: Identifier: "clang" [65:11 - 65:16] Namespace=clang:65:11 (Definition)
// CHECK-tokens: Punctuation: "{" [65:17 - 65:18] Namespace=clang:65:11 (Definition)
// CHECK-tokens: Keyword: "class" [66:1 - 66:6] ClassDecl=IdentifierInfo:66:7 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [66:7 - 66:21] ClassDecl=IdentifierInfo:66:7 (Definition)
// CHECK-tokens: Punctuation: "{" [66:22 - 66:23] ClassDecl=IdentifierInfo:66:7 (Definition)
// CHECK-tokens: Keyword: "public" [67:1 - 67:7] CXXAccessSpecifier=:67:1 (Definition)
// CHECK-tokens: Punctuation: ":" [67:7 - 67:8] CXXAccessSpecifier=:67:1 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [67:8 - 67:22] CXXConstructor=IdentifierInfo:67:8
// CHECK-tokens: Punctuation: "(" [67:22 - 67:23] CXXConstructor=IdentifierInfo:67:8
// CHECK-tokens: Punctuation: ")" [67:23 - 67:24] CXXConstructor=IdentifierInfo:67:8
// CHECK-tokens: Punctuation: ";" [67:24 - 67:25] ClassDecl=IdentifierInfo:66:7 (Definition)
// CHECK-tokens: Keyword: "const" [68:3 - 68:8] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Keyword: "char" [68:9 - 68:13] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Punctuation: "*" [68:14 - 68:15] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Identifier: "getNameStart" [68:15 - 68:27] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Punctuation: "(" [68:27 - 68:28] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Punctuation: ")" [68:28 - 68:29] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Keyword: "const" [68:30 - 68:35] CXXMethod=getNameStart:68:15 (Definition)
// CHECK-tokens: Punctuation: "{" [68:36 - 68:37] CompoundStmt=
// CHECK-tokens: Keyword: "typedef" [69:5 - 69:12] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Identifier: "std" [69:13 - 69:16] NamespaceRef=std:3:11
// CHECK-tokens: Punctuation: "::" [69:16 - 69:18] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Identifier: "pair" [69:18 - 69:22] TemplateRef=pair:4:44
// CHECK-tokens: Punctuation: "<" [69:23 - 69:24] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [69:25 - 69:39] TypeRef=class clang::IdentifierInfo:66:7
// CHECK-tokens: Punctuation: "," [69:39 - 69:40] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Keyword: "const" [69:41 - 69:46] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Keyword: "char" [69:47 - 69:51] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Punctuation: "*" [69:52 - 69:53] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Punctuation: ">" [69:53 - 69:54] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Identifier: "actualtype" [69:54 - 69:64] TypedefDecl=actualtype:69:54 (Definition)
// CHECK-tokens: Punctuation: ";" [69:64 - 69:65] DeclStmt=
// CHECK-tokens: Keyword: "return" [70:5 - 70:11] ReturnStmt=
// CHECK-tokens: Punctuation: "(" [70:12 - 70:13] ParenExpr=
// CHECK-tokens: Punctuation: "(" [70:13 - 70:14] CStyleCastExpr=
// CHECK-tokens: Keyword: "const" [70:14 - 70:19] CStyleCastExpr=
// CHECK-tokens: Identifier: "actualtype" [70:20 - 70:30] TypeRef=actualtype:69:54
// CHECK-tokens: Punctuation: "*" [70:31 - 70:32] CStyleCastExpr=
// CHECK-tokens: Punctuation: ")" [70:32 - 70:33] CStyleCastExpr=
// CHECK-tokens: Keyword: "this" [70:34 - 70:38] CXXThisExpr=
// CHECK-tokens: Punctuation: ")" [70:38 - 70:39] ParenExpr=
// CHECK-tokens: Punctuation: "->" [70:39 - 70:41] MemberRefExpr=second:4:55
// CHECK-tokens: Identifier: "second" [70:41 - 70:47] MemberRefExpr=second:4:55
// CHECK-tokens: Punctuation: ";" [70:47 - 70:48] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [71:3 - 71:4] CompoundStmt=
// CHECK-tokens: Keyword: "unsigned" [72:3 - 72:11] CXXMethod=getLength:72:12 (Definition)
// CHECK-tokens: Identifier: "getLength" [72:12 - 72:21] CXXMethod=getLength:72:12 (Definition)
// CHECK-tokens: Punctuation: "(" [72:21 - 72:22] CXXMethod=getLength:72:12 (Definition)
// CHECK-tokens: Punctuation: ")" [72:22 - 72:23] CXXMethod=getLength:72:12 (Definition)
// CHECK-tokens: Keyword: "const" [72:24 - 72:29] CXXMethod=getLength:72:12 (Definition)
// CHECK-tokens: Punctuation: "{" [72:30 - 72:31] CompoundStmt=
// CHECK-tokens: Keyword: "typedef" [73:5 - 73:12] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Identifier: "std" [73:13 - 73:16] NamespaceRef=std:3:11
// CHECK-tokens: Punctuation: "::" [73:16 - 73:18] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Identifier: "pair" [73:18 - 73:22] TemplateRef=pair:4:44
// CHECK-tokens: Punctuation: "<" [73:23 - 73:24] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [73:25 - 73:39] TypeRef=class clang::IdentifierInfo:66:7
// CHECK-tokens: Punctuation: "," [73:39 - 73:40] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Keyword: "const" [73:41 - 73:46] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Keyword: "char" [73:47 - 73:51] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Punctuation: "*" [73:52 - 73:53] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Punctuation: ">" [73:53 - 73:54] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Identifier: "actualtype" [73:54 - 73:64] TypedefDecl=actualtype:73:54 (Definition)
// CHECK-tokens: Punctuation: ";" [73:64 - 73:65] DeclStmt=
// CHECK-tokens: Keyword: "const" [74:5 - 74:10] VarDecl=p:74:17 (Definition)
// CHECK-tokens: Keyword: "char" [74:11 - 74:15] VarDecl=p:74:17 (Definition)
// CHECK-tokens: Punctuation: "*" [74:16 - 74:17] VarDecl=p:74:17 (Definition)
// CHECK-tokens: Identifier: "p" [74:17 - 74:18] VarDecl=p:74:17 (Definition)
// CHECK-tokens: Punctuation: "=" [74:19 - 74:20] VarDecl=p:74:17 (Definition)
// CHECK-tokens: Punctuation: "(" [74:21 - 74:22] ParenExpr=
// CHECK-tokens: Punctuation: "(" [74:22 - 74:23] CStyleCastExpr=
// CHECK-tokens: Keyword: "const" [74:23 - 74:28] CStyleCastExpr=
// CHECK-tokens: Identifier: "actualtype" [74:29 - 74:39] TypeRef=actualtype:73:54
// CHECK-tokens: Punctuation: "*" [74:40 - 74:41] CStyleCastExpr=
// CHECK-tokens: Punctuation: ")" [74:41 - 74:42] CStyleCastExpr=
// CHECK-tokens: Keyword: "this" [74:43 - 74:47] CXXThisExpr=
// CHECK-tokens: Punctuation: ")" [74:47 - 74:48] ParenExpr=
// CHECK-tokens: Punctuation: "->" [74:48 - 74:50] MemberRefExpr=second:4:55
// CHECK-tokens: Identifier: "second" [74:50 - 74:56] MemberRefExpr=second:4:55
// CHECK-tokens: Punctuation: "-" [74:57 - 74:58] BinaryOperator=
// CHECK-tokens: Literal: "2" [74:59 - 74:60] IntegerLiteral=
// CHECK-tokens: Punctuation: ";" [74:60 - 74:61] DeclStmt=
// CHECK-tokens: Keyword: "return" [75:5 - 75:11] ReturnStmt=
// CHECK-tokens: Punctuation: "(" [75:12 - 75:13] ParenExpr=
// CHECK-tokens: Punctuation: "(" [75:13 - 75:14] ParenExpr=
// CHECK-tokens: Punctuation: "(" [75:14 - 75:15] CStyleCastExpr=
// CHECK-tokens: Keyword: "unsigned" [75:15 - 75:23] CStyleCastExpr=
// CHECK-tokens: Punctuation: ")" [75:23 - 75:24] CStyleCastExpr=
// CHECK-tokens: Identifier: "p" [75:25 - 75:26] DeclRefExpr=p:74:17
// CHECK-tokens: Punctuation: "[" [75:26 - 75:27] ArraySubscriptExpr=
// CHECK-tokens: Literal: "0" [75:27 - 75:28] IntegerLiteral=
// CHECK-tokens: Punctuation: "]" [75:28 - 75:29] ArraySubscriptExpr=
// CHECK-tokens: Punctuation: ")" [75:29 - 75:30] ParenExpr=
// CHECK-tokens: Punctuation: "|" [75:31 - 75:32] BinaryOperator=
// CHECK-tokens: Punctuation: "(" [75:33 - 75:34] ParenExpr=
// CHECK-tokens: Punctuation: "(" [75:34 - 75:35] ParenExpr=
// CHECK-tokens: Punctuation: "(" [75:35 - 75:36] CStyleCastExpr=
// CHECK-tokens: Keyword: "unsigned" [75:36 - 75:44] CStyleCastExpr=
// CHECK-tokens: Punctuation: ")" [75:44 - 75:45] CStyleCastExpr=
// CHECK-tokens: Identifier: "p" [75:46 - 75:47] DeclRefExpr=p:74:17
// CHECK-tokens: Punctuation: "[" [75:47 - 75:48] ArraySubscriptExpr=
// CHECK-tokens: Literal: "1" [75:48 - 75:49] IntegerLiteral=
// CHECK-tokens: Punctuation: "]" [75:49 - 75:50] ArraySubscriptExpr=
// CHECK-tokens: Punctuation: ")" [75:50 - 75:51] ParenExpr=
// CHECK-tokens: Punctuation: "<<" [75:52 - 75:54] BinaryOperator=
// CHECK-tokens: Literal: "8" [75:55 - 75:56] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [75:56 - 75:57] ParenExpr=
// CHECK-tokens: Punctuation: ")" [75:57 - 75:58] ParenExpr=
// CHECK-tokens: Punctuation: "-" [75:59 - 75:60] BinaryOperator=
// CHECK-tokens: Literal: "1" [75:61 - 75:62] IntegerLiteral=
// CHECK-tokens: Punctuation: ";" [75:62 - 75:63] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [76:3 - 76:4] CompoundStmt=
// CHECK-tokens: Identifier: "llvm" [77:3 - 77:7] NamespaceRef=llvm:37:11
// CHECK-tokens: Punctuation: "::" [77:7 - 77:9] CXXMethod=getName:77:19 (Definition)
// CHECK-tokens: Identifier: "StringRef" [77:9 - 77:18] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "getName" [77:19 - 77:26] CXXMethod=getName:77:19 (Definition)
// CHECK-tokens: Punctuation: "(" [77:26 - 77:27] CXXMethod=getName:77:19 (Definition)
// CHECK-tokens: Punctuation: ")" [77:27 - 77:28] CXXMethod=getName:77:19 (Definition)
// CHECK-tokens: Keyword: "const" [77:29 - 77:34] CXXMethod=getName:77:19 (Definition)
// CHECK-tokens: Punctuation: "{" [77:35 - 77:36] CompoundStmt=
// CHECK-tokens: Keyword: "return" [78:5 - 78:11] ReturnStmt=
// CHECK-tokens: Identifier: "llvm" [78:12 - 78:16] NamespaceRef=llvm:37:11
// CHECK-tokens: Punctuation: "::" [78:16 - 78:18] CallExpr=StringRef:49:3
// CHECK-tokens: Identifier: "StringRef" [78:18 - 78:27] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Punctuation: "(" [78:27 - 78:28] CallExpr=StringRef:49:3
// CHECK-tokens: Identifier: "getNameStart" [78:28 - 78:40] MemberRefExpr=getNameStart:68:15
// CHECK-tokens: Punctuation: "(" [78:40 - 78:41] CallExpr=getNameStart:68:15
// CHECK-tokens: Punctuation: ")" [78:41 - 78:42] CallExpr=getNameStart:68:15
// CHECK-tokens: Punctuation: "," [78:42 - 78:43] CallExpr=StringRef:49:3
// CHECK-tokens: Identifier: "getLength" [78:44 - 78:53]  MemberRefExpr=getLength:72:12
// CHECK-tokens: Punctuation: "(" [78:53 - 78:54] CallExpr=getLength:72:12
// CHECK-tokens: Punctuation: ")" [78:54 - 78:55] CallExpr=getLength:72:12
// CHECK-tokens: Punctuation: ")" [78:55 - 78:56] CallExpr=StringRef:49:3
// CHECK-tokens: Punctuation: ";" [78:56 - 78:57] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [79:3 - 79:4] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [80:1 - 80:2] ClassDecl=IdentifierInfo:66:7 (Definition)
// CHECK-tokens: Punctuation: ";" [80:2 - 80:3] Namespace=clang:65:11 (Definition)
// CHECK-tokens: Punctuation: "}" [81:1 - 81:2] Namespace=clang:65:11 (Definition)
// CHECK-tokens: Keyword: "namespace" [82:1 - 82:10]
// CHECK-tokens: Identifier: "llvm" [82:11 - 82:15] Namespace=llvm:82:11 (Definition)
// CHECK-tokens: Punctuation: "{" [82:16 - 82:17] Namespace=llvm:82:11 (Definition)
// CHECK-tokens: Keyword: "template" [83:1 - 83:9] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Punctuation: "<" [83:10 - 83:11] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Keyword: "typename" [83:12 - 83:20] TemplateTypeParameter=T:83:21 (Definition)
// CHECK-tokens: Identifier: "T" [83:21 - 83:22] TemplateTypeParameter=T:83:21 (Definition)
// CHECK-tokens: Punctuation: "," [83:22 - 83:23] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Keyword: "typename" [83:24 - 83:32] TemplateTypeParameter=R:83:33 (Definition)
// CHECK-tokens: Identifier: "R" [83:33 - 83:34] TemplateTypeParameter=R:83:33 (Definition)
// CHECK-tokens: Punctuation: "=" [83:35 - 83:36] TemplateTypeParameter=R:83:33 (Definition)
// CHECK-tokens: Identifier: "T" [83:37 - 83:38] TypeRef=T:83:21
// CHECK-tokens: Punctuation: ">" [83:39 - 83:40] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Keyword: "class" [83:41 - 83:46] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Identifier: "StringSwitch" [83:47 - 83:59] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Punctuation: "{" [83:60 - 83:61] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Identifier: "StringRef" [84:3 - 84:12] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "Str" [84:13 - 84:16] FieldDecl=Str:84:13 (Definition)
// CHECK-tokens: Punctuation: ";" [84:16 - 84:17] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Keyword: "const" [85:3 - 85:8] FieldDecl=Result:85:12 (Definition)
// CHECK-tokens: Identifier: "T" [85:9 - 85:10] TypeRef=T:83:21
// CHECK-tokens: Punctuation: "*" [85:11 - 85:12] FieldDecl=Result:85:12 (Definition)
// CHECK-tokens: Identifier: "Result" [85:12 - 85:18] FieldDecl=Result:85:12 (Definition)
// CHECK-tokens: Punctuation: ";" [85:18 - 85:19] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Keyword: "public" [86:1 - 86:7] CXXAccessSpecifier=:86:1 (Definition)
// CHECK-tokens: Punctuation: ":" [86:7 - 86:8] CXXAccessSpecifier=:86:1 (Definition)
// CHECK-tokens: Keyword: "explicit" [87:3 - 87:11] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Identifier: "StringSwitch" [87:12 - 87:24] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Punctuation: "(" [87:24 - 87:25] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Identifier: "StringRef" [87:25 - 87:34] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "Str" [87:35 - 87:38] ParmDecl=Str:87:35 (Definition)
// CHECK-tokens: Punctuation: ")" [87:38 - 87:39] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Punctuation: ":" [87:40 - 87:41] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Identifier: "Str" [87:42 - 87:45] MemberRef=Str:84:13
// CHECK-tokens: Punctuation: "(" [87:45 - 87:46] CallExpr=StringRef:38:7
// CHECK-tokens: Identifier: "Str" [87:46 - 87:49] DeclRefExpr=Str:87:35
// CHECK-tokens: Punctuation: ")" [87:49 - 87:50] CallExpr=StringRef:38:7
// CHECK-tokens: Punctuation: "," [87:50 - 87:51] CXXConstructor=StringSwitch<T, R>:87:12 (Definition)
// CHECK-tokens: Identifier: "Result" [87:52 - 87:58] MemberRef=Result:85:12
// CHECK-tokens: Punctuation: "(" [87:58 - 87:59] UnexposedExpr=
// CHECK-tokens: Literal: "0" [87:59 - 87:60] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [87:60 - 87:61] UnexposedExpr=
// CHECK-tokens: Punctuation: "{" [87:62 - 87:63] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [87:63 - 87:64] CompoundStmt=
// CHECK-tokens: Keyword: "template" [88:3 - 88:11] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Punctuation: "<" [88:12 - 88:13] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Keyword: "unsigned" [88:14 - 88:22] NonTypeTemplateParameter=N:88:23 (Definition)
// CHECK-tokens: Identifier: "N" [88:23 - 88:24] NonTypeTemplateParameter=N:88:23 (Definition)
// CHECK-tokens: Punctuation: ">" [88:25 - 88:26] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Identifier: "StringSwitch" [88:27 - 88:39] TypeRef=StringSwitch<T, R>:83:47
// CHECK-tokens: Punctuation: "&" [88:40 - 88:41] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Identifier: "Case" [88:42 - 88:46] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Punctuation: "(" [88:46 - 88:47] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Keyword: "const" [88:47 - 88:52] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Keyword: "char" [88:53 - 88:57] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Punctuation: "(" [88:58 - 88:59] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Punctuation: "&" [88:59 - 88:60] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Identifier: "S" [88:60 - 88:61] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Punctuation: ")" [88:61 - 88:62] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Punctuation: "[" [88:62 - 88:63] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Identifier: "N" [88:63 - 88:64] DeclRefExpr=N:88:23
// CHECK-tokens: Punctuation: "]" [88:64 - 88:65] ParmDecl=S:88:60 (Definition)
// CHECK-tokens: Punctuation: "," [88:65 - 88:66] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Keyword: "const" [89:47 - 89:52] ParmDecl=Value:89:57 (Definition)
// CHECK-tokens: Identifier: "T" [89:53 - 89:54] TypeRef=T:83:21
// CHECK-tokens: Punctuation: "&" [89:55 - 89:56] ParmDecl=Value:89:57 (Definition)
// CHECK-tokens: Identifier: "Value" [89:57 - 89:62] ParmDecl=Value:89:57 (Definition)
// CHECK-tokens: Punctuation: ")" [89:62 - 89:63] FunctionTemplate=Case:88:42 (Definition)
// CHECK-tokens: Punctuation: "{" [89:64 - 89:65] CompoundStmt=
// CHECK-tokens: Keyword: "return" [90:5 - 90:11] ReturnStmt=
// CHECK-tokens: Punctuation: "*" [90:12 - 90:13] UnaryOperator=
// CHECK-tokens: Keyword: "this" [90:13 - 90:17] CXXThisExpr=
// CHECK-tokens: Punctuation: ";" [90:17 - 90:18] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [91:3 - 91:4] CompoundStmt=
// CHECK-tokens: Identifier: "R" [92:3 - 92:4] TypeRef=R:83:33
// CHECK-tokens: Identifier: "Default" [92:5 - 92:12] CXXMethod=Default:92:5 (Definition)
// CHECK-tokens: Punctuation: "(" [92:12 - 92:13] CXXMethod=Default:92:5 (Definition)
// CHECK-tokens: Keyword: "const" [92:13 - 92:18] ParmDecl=Value:92:23 (Definition)
// CHECK-tokens: Identifier: "T" [92:19 - 92:20] TypeRef=T:83:21
// CHECK-tokens: Punctuation: "&" [92:21 - 92:22] ParmDecl=Value:92:23 (Definition)
// CHECK-tokens: Identifier: "Value" [92:23 - 92:28] ParmDecl=Value:92:23 (Definition)
// CHECK-tokens: Punctuation: ")" [92:28 - 92:29] CXXMethod=Default:92:5 (Definition)
// CHECK-tokens: Keyword: "const" [92:30 - 92:35] CXXMethod=Default:92:5 (Definition)
// CHECK-tokens: Punctuation: "{" [92:36 - 92:37] CompoundStmt=
// CHECK-tokens: Keyword: "return" [93:5 - 93:11] ReturnStmt=
// CHECK-tokens: Identifier: "Value" [93:12 - 93:17] DeclRefExpr=Value:92:23
// CHECK-tokens: Punctuation: ";" [93:17 - 93:18] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [94:3 - 94:4] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [95:1 - 95:2] ClassTemplate=StringSwitch:83:47 (Definition)
// CHECK-tokens: Punctuation: ";" [95:2 - 95:3] Namespace=llvm:82:11 (Definition)
// CHECK-tokens: Punctuation: "}" [96:1 - 96:2] Namespace=llvm:82:11 (Definition)
// CHECK-tokens: Keyword: "using" [98:1 - 98:6] UsingDirective=:98:17
// CHECK-tokens: Keyword: "namespace" [98:7 - 98:16] UsingDirective=:98:17
// CHECK-tokens: Identifier: "clang" [98:17 - 98:22] NamespaceRef=clang:10:17
// CHECK-tokens: Punctuation: ";" [98:22 - 98:23]
// CHECK-tokens: Identifier: "AttributeList" [100:1 - 100:14] TypeRef=class clang::AttributeList:12:9
// CHECK-tokens: Punctuation: "::" [100:14 - 100:16] CXXMethod=getKind:100:36 (Definition) (static)
// CHECK-tokens: Identifier: "Kind" [100:16 - 100:20] TypeRef=enum clang::AttributeList::Kind:13:10
// CHECK-tokens: Identifier: "AttributeList" [100:21 - 100:34] TypeRef=class clang::AttributeList:12:9
// CHECK-tokens: Punctuation: "::" [100:34 - 100:36] CXXMethod=getKind:100:36 (Definition) (static)
// CHECK-tokens: Identifier: "getKind" [100:36 - 100:43] CXXMethod=getKind:100:36 (Definition) (static)
// CHECK-tokens: Punctuation: "(" [100:43 - 100:44] CXXMethod=getKind:100:36 (Definition) (static)
// CHECK-tokens: Keyword: "const" [100:44 - 100:49] ParmDecl=Name:100:67 (Definition)
// CHECK-tokens: Identifier: "IdentifierInfo" [100:50 - 100:64] TypeRef=class clang::IdentifierInfo:66:7
// CHECK-tokens: Punctuation: "*" [100:65 - 100:66] ParmDecl=Name:100:67 (Definition)
// CHECK-tokens: Identifier: "Name" [100:67 - 100:71] ParmDecl=Name:100:67 (Definition)
// CHECK-tokens: Punctuation: ")" [100:71 - 100:72] CXXMethod=getKind:100:36 (Definition) (static)
// CHECK-tokens: Punctuation: "{" [100:73 - 100:74] CompoundStmt=
// CHECK-tokens: Identifier: "llvm" [101:3 - 101:7] NamespaceRef=llvm:82:11
// CHECK-tokens: Punctuation: "::" [101:7 - 101:9] VarDecl=AttrName:101:19 (Definition)
// CHECK-tokens: Identifier: "StringRef" [101:9 - 101:18] TypeRef=class llvm::StringRef:38:7
// CHECK-tokens: Identifier: "AttrName" [101:19 - 101:27] VarDecl=AttrName:101:19 (Definition)
// CHECK-tokens: Punctuation: "=" [101:28 - 101:29] VarDecl=AttrName:101:19 (Definition)
// CHECK-tokens: Identifier: "Name" [101:30 - 101:34] DeclRefExpr=Name:100:67
// CHECK-tokens: Punctuation: "->" [101:34 - 101:36] MemberRefExpr=getName:77:19
// CHECK-tokens: Identifier: "getName" [101:36 - 101:43] MemberRefExpr=getName:77:19
// CHECK-tokens: Punctuation: "(" [101:43 - 101:44] CallExpr=getName:77:19
// CHECK-tokens: Punctuation: ")" [101:44 - 101:45] CallExpr=getName:77:19
// CHECK-tokens: Punctuation: ";" [101:45 - 101:46] DeclStmt=
// CHECK-tokens: Keyword: "if" [102:3 - 102:5] IfStmt=
// CHECK-tokens: Punctuation: "(" [102:6 - 102:7] IfStmt=
// CHECK-tokens: Identifier: "AttrName" [102:7 - 102:15] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: "." [102:15 - 102:16] MemberRefExpr=startswith:52:8
// CHECK-tokens: Identifier: "startswith" [102:16 - 102:26] MemberRefExpr=startswith:52:8
// CHECK-tokens: Punctuation: "(" [102:26 - 102:27] CallExpr=startswith:52:8
// CHECK-tokens: Literal: ""__"" [102:27 - 102:31] StringLiteral=
// CHECK-tokens: Punctuation: ")" [102:31 - 102:32] CallExpr=startswith:52:8
// CHECK-tokens: Punctuation: "&&" [102:33 - 102:35] BinaryOperator=
// CHECK-tokens: Identifier: "AttrName" [102:36 - 102:44] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: "." [102:44 - 102:45] MemberRefExpr=endswith:56:8
// CHECK-tokens: Identifier: "endswith" [102:45 - 102:53] MemberRefExpr=endswith:56:8
// CHECK-tokens: Punctuation: "(" [102:53 - 102:54] CallExpr=endswith:56:8
// CHECK-tokens: Literal: ""__"" [102:54 - 102:58] StringLiteral=
// CHECK-tokens: Punctuation: ")" [102:58 - 102:59] CallExpr=endswith:56:8
// CHECK-tokens: Punctuation: ")" [102:59 - 102:60] IfStmt=
// CHECK-tokens: Identifier: "AttrName" [103:5 - 103:13] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: "=" [103:14 - 103:15] DeclRefExpr=operator=:38:7
// CHECK-tokens: Identifier: "AttrName" [103:16 - 103:24] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: "." [103:24 - 103:25] MemberRefExpr=substr:60:13
// CHECK-tokens: Identifier: "substr" [103:25 - 103:31] MemberRefExpr=substr:60:13
// CHECK-tokens: Punctuation: "(" [103:31 - 103:32] CallExpr=substr:60:13
// CHECK-tokens: Literal: "2" [103:32 - 103:33] IntegerLiteral=
// CHECK-tokens: Punctuation: "," [103:33 - 103:34] CallExpr=substr:60:13
// CHECK-tokens: Identifier: "AttrName" [103:35 - 103:43] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: "." [103:43 - 103:44] MemberRefExpr=size:51:10
// CHECK-tokens: Identifier: "size" [103:44 - 103:48] MemberRefExpr=size:51:10
// CHECK-tokens: Punctuation: "(" [103:48 - 103:49] CallExpr=size:51:10
// CHECK-tokens: Punctuation: ")" [103:49 - 103:50] CallExpr=size:51:10
// CHECK-tokens: Punctuation: "-" [103:51 - 103:52] BinaryOperator=
// CHECK-tokens: Literal: "4" [103:53 - 103:54] IntegerLiteral=
// CHECK-tokens: Punctuation: ")" [103:54 - 103:55] CallExpr=substr:60:13
// CHECK-tokens: Punctuation: ";" [103:55 - 103:56] CompoundStmt=
// CHECK-tokens: Keyword: "return" [105:3 - 105:9] ReturnStmt=
// FIXME: Missing "llvm" namespace reference below
// CHECK-tokens: Identifier: "llvm" [105:10 - 105:14] NamespaceRef=llvm:82:11
// CHECK-tokens: Punctuation: "::" [105:14 - 105:16] CXXFunctionalCastExpr=
// CHECK-tokens: Identifier: "StringSwitch" [105:16 - 105:28] TemplateRef=StringSwitch:83:47
// CHECK-tokens: Punctuation: "<" [105:29 - 105:30] CXXFunctionalCastExpr=
// CHECK-tokens: Identifier: "AttributeList" [105:31 - 105:44] TypeRef=class clang::AttributeList:12:9
// CHECK-tokens: Punctuation: "::" [105:44 - 105:46] CXXFunctionalCastExpr=
// CHECK-tokens: Identifier: "Kind" [105:46 - 105:50] TypeRef=enum clang::AttributeList::Kind:13:10
// CHECK-tokens: Punctuation: ">" [105:51 - 105:52] CallExpr=StringSwitch:87:12
// CHECK-tokens: Punctuation: "(" [105:53 - 105:54] CallExpr=StringSwitch:87:12
// CHECK-tokens: Identifier: "AttrName" [105:54 - 105:62] DeclRefExpr=AttrName:101:19
// CHECK-tokens: Punctuation: ")" [105:62 - 105:63] CallExpr=StringSwitch:87:12
// CHECK-tokens: Punctuation: "." [106:5 - 106:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [106:6 - 106:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [106:10 - 106:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""weak"" [106:11 - 106:17] StringLiteral=
// CHECK-tokens: Punctuation: "," [106:17 - 106:18] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_weak" [106:19 - 106:26] DeclRefExpr=AT_weak:29:45
// CHECK-tokens: Punctuation: ")" [106:26 - 106:27] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [107:5 - 107:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [107:6 - 107:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [107:10 - 107:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""weakref"" [107:11 - 107:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [107:20 - 107:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_weakref" [107:22 - 107:32] DeclRefExpr=AT_weakref:29:54
// CHECK-tokens: Punctuation: ")" [107:32 - 107:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [108:5 - 108:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [108:6 - 108:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [108:10 - 108:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""pure"" [108:11 - 108:17] StringLiteral=
// CHECK-tokens: Punctuation: "," [108:17 - 108:18] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_pure" [108:19 - 108:26] DeclRefExpr=AT_pure:26:49
// CHECK-tokens: Punctuation: ")" [108:26 - 108:27] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [109:5 - 109:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [109:6 - 109:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [109:10 - 109:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""mode"" [109:11 - 109:17] StringLiteral=
// CHECK-tokens: Punctuation: "," [109:17 - 109:18] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_mode" [109:19 - 109:26] DeclRefExpr=AT_mode:20:44
// CHECK-tokens: Punctuation: ")" [109:26 - 109:27] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [110:5 - 110:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [110:6 - 110:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [110:10 - 110:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""used"" [110:11 - 110:17] StringLiteral=
// CHECK-tokens: Punctuation: "," [110:17 - 110:18] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_used" [110:19 - 110:26] DeclRefExpr=AT_used:28:34
// CHECK-tokens: Punctuation: ")" [110:26 - 110:27] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [111:5 - 111:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [111:6 - 111:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [111:10 - 111:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""alias"" [111:11 - 111:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [111:18 - 111:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_alias" [111:20 - 111:28] DeclRefExpr=AT_alias:15:25
// CHECK-tokens: Punctuation: ")" [111:28 - 111:29] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [112:5 - 112:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [112:6 - 112:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [112:10 - 112:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""align"" [112:11 - 112:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [112:18 - 112:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_aligned" [112:20 - 112:30] DeclRefExpr=AT_aligned:15:35
// CHECK-tokens: Punctuation: ")" [112:30 - 112:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [113:5 - 113:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [113:6 - 113:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [113:10 - 113:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""final"" [113:11 - 113:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [113:18 - 113:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_final" [113:20 - 113:28] DeclRefExpr=AT_final:19:40
// CHECK-tokens: Punctuation: ")" [113:28 - 113:29] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [114:5 - 114:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [114:6 - 114:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [114:10 - 114:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""cdecl"" [114:11 - 114:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [114:18 - 114:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_cdecl" [114:20 - 114:28] DeclRefExpr=AT_cdecl:17:30
// CHECK-tokens: Punctuation: ")" [114:28 - 114:29] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [115:5 - 115:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [115:6 - 115:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [115:10 - 115:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""const"" [115:11 - 115:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [115:18 - 115:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_const" [115:20 - 115:28] DeclRefExpr=AT_const:17:52
// CHECK-tokens: Punctuation: ")" [115:28 - 115:29] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [116:5 - 116:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [116:6 - 116:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [116:10 - 116:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__const"" [116:11 - 116:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [116:20 - 116:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_const" [116:22 - 116:30] DeclRefExpr=AT_const:17:52
// CHECK-tokens: Punctuation: ")" [116:30 - 116:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [117:5 - 117:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [117:6 - 117:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [117:10 - 117:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""blocks"" [117:11 - 117:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [117:19 - 117:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_blocks" [117:21 - 117:30] DeclRefExpr=AT_blocks:16:57
// CHECK-tokens: Punctuation: ")" [117:30 - 117:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [118:5 - 118:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [118:6 - 118:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [118:10 - 118:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""format"" [118:11 - 118:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [118:19 - 118:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_format" [118:21 - 118:30] DeclRefExpr=AT_format:19:50
// CHECK-tokens: Punctuation: ")" [118:30 - 118:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [119:5 - 119:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [119:6 - 119:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [119:10 - 119:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""hiding"" [119:11 - 119:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [119:19 - 119:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_hiding" [119:21 - 119:30] DeclRefExpr=AT_hiding:20:22
// CHECK-tokens: Punctuation: ")" [119:30 - 119:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [120:5 - 120:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [120:6 - 120:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [120:10 - 120:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""malloc"" [120:11 - 120:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [120:19 - 120:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_malloc" [120:21 - 120:30] DeclRefExpr=AT_malloc:20:33
// CHECK-tokens: Punctuation: ")" [120:30 - 120:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [121:5 - 121:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [121:6 - 121:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [121:10 - 121:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""packed"" [121:11 - 121:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [121:19 - 121:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_packed" [121:21 - 121:30] DeclRefExpr=AT_packed:26:27
// CHECK-tokens: Punctuation: ")" [121:30 - 121:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [122:5 - 122:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [122:6 - 122:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [122:10 - 122:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""unused"" [122:11 - 122:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [122:19 - 122:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_unused" [122:21 - 122:30] DeclRefExpr=AT_unused:28:23
// CHECK-tokens: Punctuation: ")" [122:30 - 122:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [123:5 - 123:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [123:6 - 123:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [123:10 - 123:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""aligned"" [123:11 - 123:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [123:20 - 123:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_aligned" [123:22 - 123:32] DeclRefExpr=AT_aligned:15:35
// CHECK-tokens: Punctuation: ")" [123:32 - 123:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [124:5 - 124:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [124:6 - 124:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [124:10 - 124:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""cleanup"" [124:11 - 124:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [124:20 - 124:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_cleanup" [124:22 - 124:32] DeclRefExpr=AT_cleanup:17:40
// CHECK-tokens: Punctuation: ")" [124:32 - 124:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [125:5 - 125:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [125:6 - 125:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [125:10 - 125:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""naked"" [125:11 - 125:18] StringLiteral=
// CHECK-tokens: Punctuation: "," [125:18 - 125:19] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_naked" [125:20 - 125:28] DeclRefExpr=AT_naked:20:53
// CHECK-tokens: Punctuation: ")" [125:28 - 125:29] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [126:5 - 126:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [126:6 - 126:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [126:10 - 126:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""nodebug"" [126:11 - 126:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [126:20 - 126:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_nodebug" [126:22 - 126:32] DeclRefExpr=AT_nodebug:20:63
// CHECK-tokens: Punctuation: ")" [126:32 - 126:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [127:5 - 127:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [127:6 - 127:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [127:10 - 127:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""nonnull"" [127:11 - 127:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [127:20 - 127:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_nonnull" [127:22 - 127:32] DeclRefExpr=AT_nonnull:21:47
// CHECK-tokens: Punctuation: ")" [127:32 - 127:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [128:5 - 128:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [128:6 - 128:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [128:10 - 128:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""nothrow"" [128:11 - 128:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [128:20 - 128:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_nothrow" [128:22 - 128:32] DeclRefExpr=AT_nothrow:22:7
// CHECK-tokens: Punctuation: ")" [128:32 - 128:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [129:5 - 129:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [129:6 - 129:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [129:10 - 129:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""objc_gc"" [129:11 - 129:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [129:20 - 129:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_objc_gc" [129:22 - 129:32] DeclRefExpr=AT_objc_gc:24:59
// CHECK-tokens: Punctuation: ")" [129:32 - 129:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [130:5 - 130:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [130:6 - 130:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [130:10 - 130:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""regparm"" [130:11 - 130:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [130:20 - 130:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_regparm" [130:22 - 130:32] DeclRefExpr=AT_regparm:26:58
// CHECK-tokens: Punctuation: ")" [130:32 - 130:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [131:5 - 131:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [131:6 - 131:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [131:10 - 131:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""section"" [131:11 - 131:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [131:20 - 131:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_section" [131:22 - 131:32] DeclRefExpr=AT_section:27:7
// CHECK-tokens: Punctuation: ")" [131:32 - 131:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [132:5 - 132:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [132:6 - 132:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [132:10 - 132:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""stdcall"" [132:11 - 132:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [132:20 - 132:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_stdcall" [132:22 - 132:32] DeclRefExpr=AT_stdcall:27:32
// CHECK-tokens: Punctuation: ")" [132:32 - 132:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [133:5 - 133:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [133:6 - 133:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [133:10 - 133:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""annotate"" [133:11 - 133:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [133:21 - 133:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_annotate" [133:23 - 133:34] DeclRefExpr=AT_annotate:16:29
// CHECK-tokens: Punctuation: ")" [133:34 - 133:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [134:5 - 134:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [134:6 - 134:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [134:10 - 134:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""fastcall"" [134:11 - 134:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [134:21 - 134:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_fastcall" [134:23 - 134:34] DeclRefExpr=AT_fastcall:19:27
// CHECK-tokens: Punctuation: ")" [134:34 - 134:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [135:5 - 135:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [135:6 - 135:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [135:10 - 135:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ibaction"" [135:11 - 135:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [135:21 - 135:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_IBAction" [135:23 - 135:34] DeclRefExpr=AT_IBAction:14:7
// CHECK-tokens: Punctuation: ")" [135:34 - 135:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [136:5 - 136:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [136:6 - 136:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [136:10 - 136:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""iboutlet"" [136:11 - 136:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [136:21 - 136:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_IBOutlet" [136:23 - 136:34] DeclRefExpr=AT_IBOutlet:14:20
// CHECK-tokens: Punctuation: ")" [136:34 - 136:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [137:5 - 137:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [137:6 - 137:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [137:10 - 137:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""iboutletcollection"" [137:11 - 137:31] StringLiteral=
// CHECK-tokens: Punctuation: "," [137:31 - 137:32] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_IBOutletCollection" [137:33 - 137:54] DeclRefExpr=AT_IBOutletCollection:14:33
// CHECK-tokens: Punctuation: ")" [137:54 - 137:55] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [138:5 - 138:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [138:6 - 138:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [138:10 - 138:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""noreturn"" [138:11 - 138:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [138:21 - 138:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_noreturn" [138:23 - 138:34] DeclRefExpr=AT_noreturn:21:59
// CHECK-tokens: Punctuation: ")" [138:34 - 138:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [139:5 - 139:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [139:6 - 139:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [139:10 - 139:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""noinline"" [139:11 - 139:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [139:21 - 139:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_noinline" [139:23 - 139:34] DeclRefExpr=AT_noinline:21:7
// CHECK-tokens: Punctuation: ")" [139:34 - 139:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [140:5 - 140:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [140:6 - 140:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [140:10 - 140:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""override"" [140:11 - 140:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [140:21 - 140:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_override" [140:23 - 140:34] DeclRefExpr=AT_override:22:51
// CHECK-tokens: Punctuation: ")" [140:34 - 140:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [141:5 - 141:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [141:6 - 141:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [141:10 - 141:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""sentinel"" [141:11 - 141:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [141:21 - 141:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_sentinel" [141:23 - 141:34] DeclRefExpr=AT_sentinel:27:19
// CHECK-tokens: Punctuation: ")" [141:34 - 141:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [142:5 - 142:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [142:6 - 142:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [142:10 - 142:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""NSObject"" [142:11 - 142:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [142:21 - 142:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_nsobject" [142:23 - 142:34] DeclRefExpr=AT_nsobject:22:19
// CHECK-tokens: Punctuation: ")" [142:34 - 142:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [143:5 - 143:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [143:6 - 143:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [143:10 - 143:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""dllimport"" [143:11 - 143:22] StringLiteral=
// CHECK-tokens: Punctuation: "," [143:22 - 143:23] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_dllimport" [143:24 - 143:36] DeclRefExpr=AT_dllimport:18:51
// CHECK-tokens: Punctuation: ")" [143:36 - 143:37] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [144:5 - 144:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [144:6 - 144:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [144:10 - 144:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""dllexport"" [144:11 - 144:22] StringLiteral=
// CHECK-tokens: Punctuation: "," [144:22 - 144:23] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_dllexport" [144:24 - 144:36] DeclRefExpr=AT_dllexport:18:37
// CHECK-tokens: Punctuation: ")" [144:36 - 144:37] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [145:5 - 145:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [145:6 - 145:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [145:10 - 145:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""may_alias"" [145:11 - 145:22] StringLiteral=
// CHECK-tokens: Punctuation: "," [145:22 - 145:23] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "IgnoredAttribute" [145:24 - 145:40] DeclRefExpr=IgnoredAttribute:31:25
// CHECK-tokens: Punctuation: ")" [145:40 - 145:41] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [146:5 - 146:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [146:6 - 146:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [146:10 - 146:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""base_check"" [146:11 - 146:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [146:23 - 146:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_base_check" [146:25 - 146:38] DeclRefExpr=AT_base_check:16:42
// CHECK-tokens: Punctuation: ")" [146:38 - 146:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [147:5 - 147:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [147:6 - 147:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [147:10 - 147:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""deprecated"" [147:11 - 147:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [147:23 - 147:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_deprecated" [147:25 - 147:38] DeclRefExpr=AT_deprecated:18:7
// CHECK-tokens: Punctuation: ")" [147:38 - 147:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [148:5 - 148:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [148:6 - 148:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [148:10 - 148:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""visibility"" [148:11 - 148:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [148:23 - 148:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_visibility" [148:25 - 148:38] DeclRefExpr=AT_visibility:29:7
// CHECK-tokens: Punctuation: ")" [148:38 - 148:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [149:5 - 149:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [149:6 - 149:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [149:10 - 149:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""destructor"" [149:11 - 149:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [149:23 - 149:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_destructor" [149:25 - 149:38] DeclRefExpr=AT_destructor:18:22
// CHECK-tokens: Punctuation: ")" [149:38 - 149:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [150:5 - 150:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [150:6 - 150:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [150:10 - 150:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""format_arg"" [150:11 - 150:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [150:23 - 150:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_format_arg" [150:25 - 150:38] DeclRefExpr=AT_format_arg:19:61
// CHECK-tokens: Punctuation: ")" [150:38 - 150:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [151:5 - 151:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [151:6 - 151:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [151:10 - 151:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""gnu_inline"" [151:11 - 151:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [151:23 - 151:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_gnu_inline" [151:25 - 151:38] DeclRefExpr=AT_gnu_inline:20:7
// CHECK-tokens: Punctuation: ")" [151:38 - 151:39] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [152:5 - 152:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [152:6 - 152:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [152:10 - 152:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""weak_import"" [152:11 - 152:24] StringLiteral=
// CHECK-tokens: Punctuation: "," [152:24 - 152:25] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_weak_import" [152:26 - 152:40] DeclRefExpr=AT_weak_import:30:7
// CHECK-tokens: Punctuation: ")" [152:40 - 152:41] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [153:5 - 153:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [153:6 - 153:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [153:10 - 153:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""vecreturn"" [153:11 - 153:22] StringLiteral=
// CHECK-tokens: Punctuation: "," [153:22 - 153:23] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_vecreturn" [153:24 - 153:36] DeclRefExpr=AT_vecreturn:28:43
// CHECK-tokens: Punctuation: ")" [153:36 - 153:37] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [154:5 - 154:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [154:6 - 154:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [154:10 - 154:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""vector_size"" [154:11 - 154:24] StringLiteral=
// CHECK-tokens: Punctuation: "," [154:24 - 154:25] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_vector_size" [154:26 - 154:40] DeclRefExpr=AT_vector_size:28:57
// CHECK-tokens: Punctuation: ")" [154:40 - 154:41] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [155:5 - 155:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [155:6 - 155:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [155:10 - 155:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""constructor"" [155:11 - 155:24] StringLiteral=
// CHECK-tokens: Punctuation: "," [155:24 - 155:25] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_constructor" [155:26 - 155:40] DeclRefExpr=AT_constructor:17:62
// CHECK-tokens: Punctuation: ")" [155:40 - 155:41] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [156:5 - 156:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [156:6 - 156:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [156:10 - 156:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""unavailable"" [156:11 - 156:24] StringLiteral=
// CHECK-tokens: Punctuation: "," [156:24 - 156:25] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_unavailable" [156:26 - 156:40] DeclRefExpr=AT_unavailable:28:7
// CHECK-tokens: Punctuation: ")" [156:40 - 156:41] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [157:5 - 157:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [157:6 - 157:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [157:10 - 157:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""overloadable"" [157:11 - 157:25] StringLiteral=
// CHECK-tokens: Punctuation: "," [157:25 - 157:26] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_overloadable" [157:27 - 157:42] DeclRefExpr=AT_overloadable:25:7
// CHECK-tokens: Punctuation: ")" [157:42 - 157:43] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [158:5 - 158:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [158:6 - 158:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [158:10 - 158:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""address_space"" [158:11 - 158:26] StringLiteral=
// CHECK-tokens: Punctuation: "," [158:26 - 158:27] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_address_space" [158:28 - 158:44] DeclRefExpr=AT_address_space:15:7
// CHECK-tokens: Punctuation: ")" [158:44 - 158:45] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [159:5 - 159:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [159:6 - 159:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [159:10 - 159:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""always_inline"" [159:11 - 159:26] StringLiteral=
// CHECK-tokens: Punctuation: "," [159:26 - 159:27] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_always_inline" [159:28 - 159:44] DeclRefExpr=AT_always_inline:15:47
// CHECK-tokens: Punctuation: ")" [159:44 - 159:45] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [160:5 - 160:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [160:6 - 160:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [160:10 - 160:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""returns_twice"" [160:11 - 160:26] StringLiteral=
// CHECK-tokens: Punctuation: "," [160:26 - 160:27] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_returns_twice" [160:28 - 160:44] DeclRefExpr=AT_returns_twice:31:7
// CHECK-tokens: Punctuation: ")" [160:44 - 160:45] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [161:5 - 161:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [161:6 - 161:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [161:10 - 161:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""vec_type_hint"" [161:11 - 161:26] StringLiteral=
// CHECK-tokens: Punctuation: "," [161:26 - 161:27] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "IgnoredAttribute" [161:28 - 161:44] DeclRefExpr=IgnoredAttribute:31:25
// CHECK-tokens: Punctuation: ")" [161:44 - 161:45] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [162:5 - 162:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [162:6 - 162:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [162:10 - 162:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""objc_exception"" [162:11 - 162:27] StringLiteral=
// CHECK-tokens: Punctuation: "," [162:27 - 162:28] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_objc_exception" [162:29 - 162:46] DeclRefExpr=AT_objc_exception:22:32
// CHECK-tokens: Punctuation: ")" [162:46 - 162:47] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [163:5 - 163:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [163:6 - 163:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [163:10 - 163:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ext_vector_type"" [163:11 - 163:28] StringLiteral=
// CHECK-tokens: Punctuation: "," [163:28 - 163:29] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ext_vector_type" [163:30 - 163:48] DeclRefExpr=AT_ext_vector_type:19:7
// CHECK-tokens: Punctuation: ")" [163:48 - 163:49] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [164:5 - 164:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [164:6 - 164:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [164:10 - 164:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""transparent_union"" [164:11 - 164:30] StringLiteral=
// CHECK-tokens: Punctuation: "," [164:30 - 164:31] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_transparent_union" [164:32 - 164:52] DeclRefExpr=AT_transparent_union:27:57
// CHECK-tokens: Punctuation: ")" [164:52 - 164:53] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [165:5 - 165:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [165:6 - 165:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [165:10 - 165:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""analyzer_noreturn"" [165:11 - 165:30] StringLiteral=
// CHECK-tokens: Punctuation: "," [165:30 - 165:31] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_analyzer_noreturn" [165:32 - 165:52] DeclRefExpr=AT_analyzer_noreturn:16:7
// CHECK-tokens: Punctuation: ")" [165:52 - 165:53] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [166:5 - 166:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [166:6 - 166:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [166:10 - 166:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""warn_unused_result"" [166:11 - 166:31] StringLiteral=
// CHECK-tokens: Punctuation: "," [166:31 - 166:32] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_warn_unused_result" [166:33 - 166:54] DeclRefExpr=AT_warn_unused_result:29:22
// CHECK-tokens: Punctuation: ")" [166:54 - 166:55] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [167:5 - 167:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [167:6 - 167:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [167:10 - 167:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""carries_dependency"" [167:11 - 167:31] StringLiteral=
// CHECK-tokens: Punctuation: "," [167:31 - 167:32] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_carries_dependency" [167:33 - 167:54] DeclRefExpr=AT_carries_dependency:17:7
// CHECK-tokens: Punctuation: ")" [167:54 - 167:55] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [168:5 - 168:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [168:6 - 168:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [168:10 - 168:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ns_returns_not_retained"" [168:11 - 168:36] StringLiteral=
// CHECK-tokens: Punctuation: "," [168:36 - 168:37] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ns_returns_not_retained" [168:38 - 168:64] DeclRefExpr=AT_ns_returns_not_retained:24:7
// CHECK-tokens: Punctuation: ")" [168:64 - 168:65] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [169:5 - 169:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [169:6 - 169:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [169:10 - 169:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ns_returns_retained"" [169:11 - 169:32] StringLiteral=
// CHECK-tokens: Punctuation: "," [169:32 - 169:33] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ns_returns_retained" [169:34 - 169:56] DeclRefExpr=AT_ns_returns_retained:24:35
// CHECK-tokens: Punctuation: ")" [169:56 - 169:57] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [170:5 - 170:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [170:6 - 170:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [170:10 - 170:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""cf_returns_not_retained"" [170:11 - 170:36] StringLiteral=
// CHECK-tokens: Punctuation: "," [170:36 - 170:37] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_cf_returns_not_retained" [170:38 - 170:64] DeclRefExpr=AT_cf_returns_not_retained:23:7
// CHECK-tokens: Punctuation: ")" [170:64 - 170:65] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [171:5 - 171:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [171:6 - 171:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [171:10 - 171:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""cf_returns_retained"" [171:11 - 171:32] StringLiteral=
// CHECK-tokens: Punctuation: "," [171:32 - 171:33] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_cf_returns_retained" [171:34 - 171:56] DeclRefExpr=AT_cf_returns_retained:23:35
// CHECK-tokens: Punctuation: ")" [171:56 - 171:57] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [172:5 - 172:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [172:6 - 172:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [172:10 - 172:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ownership_returns"" [172:11 - 172:30] StringLiteral=
// CHECK-tokens: Punctuation: "," [172:30 - 172:31] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ownership_returns" [172:32 - 172:52] DeclRefExpr=AT_ownership_returns:25:44
// CHECK-tokens: Punctuation: ")" [172:52 - 172:53] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [173:5 - 173:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [173:6 - 173:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [173:10 - 173:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ownership_holds"" [173:11 - 173:28] StringLiteral=
// CHECK-tokens: Punctuation: "," [173:28 - 173:29] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ownership_holds" [173:30 - 173:48] DeclRefExpr=AT_ownership_holds:25:24
// CHECK-tokens: Punctuation: ")" [173:48 - 173:49] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [174:5 - 174:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [174:6 - 174:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [174:10 - 174:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""ownership_takes"" [174:11 - 174:28] StringLiteral=
// CHECK-tokens: Punctuation: "," [174:28 - 174:29] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_ownership_takes" [174:30 - 174:48] DeclRefExpr=AT_ownership_takes:26:7
// CHECK-tokens: Punctuation: ")" [174:48 - 174:49] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [175:5 - 175:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [175:6 - 175:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [175:10 - 175:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""reqd_work_group_size"" [175:11 - 175:33] StringLiteral=
// CHECK-tokens: Punctuation: "," [175:33 - 175:34] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_reqd_wg_size" [175:35 - 175:50] DeclRefExpr=AT_reqd_wg_size:30:23
// CHECK-tokens: Punctuation: ")" [175:50 - 175:51] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [176:5 - 176:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [176:6 - 176:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [176:10 - 176:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""init_priority"" [176:11 - 176:26] StringLiteral=
// CHECK-tokens: Punctuation: "," [176:26 - 176:27] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_init_priority" [176:28 - 176:44] DeclRefExpr=AT_init_priority:30:40
// CHECK-tokens: Punctuation: ")" [176:44 - 176:45] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [177:5 - 177:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [177:6 - 177:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [177:10 - 177:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""no_instrument_function"" [177:11 - 177:35] StringLiteral=
// CHECK-tokens: Punctuation: "," [177:35 - 177:36] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_no_instrument_function" [177:37 - 177:62] DeclRefExpr=AT_no_instrument_function:21:20
// CHECK-tokens: Punctuation: ")" [177:62 - 177:63] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [178:5 - 178:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [178:6 - 178:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [178:10 - 178:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""thiscall"" [178:11 - 178:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [178:21 - 178:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_thiscall" [178:23 - 178:34] DeclRefExpr=AT_thiscall:27:44
// CHECK-tokens: Punctuation: ")" [178:34 - 178:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [179:5 - 179:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [179:6 - 179:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [179:10 - 179:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""pascal"" [179:11 - 179:19] StringLiteral=
// CHECK-tokens: Punctuation: "," [179:19 - 179:20] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_pascal" [179:21 - 179:30] DeclRefExpr=AT_pascal:26:38
// CHECK-tokens: Punctuation: ")" [179:30 - 179:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [180:5 - 180:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [180:6 - 180:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [180:10 - 180:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__cdecl"" [180:11 - 180:20] StringLiteral=
// CHECK-tokens: Punctuation: "," [180:20 - 180:21] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_cdecl" [180:22 - 180:30] DeclRefExpr=AT_cdecl:17:30
// CHECK-tokens: Punctuation: ")" [180:30 - 180:31] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [181:5 - 181:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [181:6 - 181:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [181:10 - 181:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__stdcall"" [181:11 - 181:22] StringLiteral=
// CHECK-tokens: Punctuation: "," [181:22 - 181:23] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_stdcall" [181:24 - 181:34] DeclRefExpr=AT_stdcall:27:32
// CHECK-tokens: Punctuation: ")" [181:34 - 181:35] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [182:5 - 182:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [182:6 - 182:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [182:10 - 182:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__fastcall"" [182:11 - 182:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [182:23 - 182:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_fastcall" [182:25 - 182:36] DeclRefExpr=AT_fastcall:19:27
// CHECK-tokens: Punctuation: ")" [182:36 - 182:37] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [183:5 - 183:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [183:6 - 183:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [183:10 - 183:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__thiscall"" [183:11 - 183:23] StringLiteral=
// CHECK-tokens: Punctuation: "," [183:23 - 183:24] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_thiscall" [183:25 - 183:36] DeclRefExpr=AT_thiscall:27:44
// CHECK-tokens: Punctuation: ")" [183:36 - 183:37] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [184:5 - 184:6] MemberRefExpr=Case:88:42
// CHECK-tokens: Identifier: "Case" [184:6 - 184:10] MemberRefExpr=Case:88:42
// CHECK-tokens: Punctuation: "(" [184:10 - 184:11] CallExpr=Case:88:42
// CHECK-tokens: Literal: ""__pascal"" [184:11 - 184:21] StringLiteral=
// CHECK-tokens: Punctuation: "," [184:21 - 184:22] CallExpr=Case:88:42
// CHECK-tokens: Identifier: "AT_pascal" [184:23 - 184:32] DeclRefExpr=AT_pascal:26:38
// CHECK-tokens: Punctuation: ")" [184:32 - 184:33] CallExpr=Case:88:42
// CHECK-tokens: Punctuation: "." [185:5 - 185:6] MemberRefExpr=Default:92:5
// CHECK-tokens: Identifier: "Default" [185:6 - 185:13] MemberRefExpr=Default:92:5
// CHECK-tokens: Punctuation: "(" [185:13 - 185:14] CallExpr=Default:92:5
// CHECK-tokens: Identifier: "UnknownAttribute" [185:14 - 185:30] DeclRefExpr=UnknownAttribute:31:43
// CHECK-tokens: Punctuation: ")" [185:30 - 185:31] CallExpr=Default:92:5
// CHECK-tokens: Punctuation: ";" [185:31 - 185:32] CompoundStmt=
// CHECK-tokens: Punctuation: "}" [186:1 - 186:2] CompoundStmt=

// RUN: c-index-test -test-load-source all %s -std=c++98 -target x86_64-unknown-unknown 2>&1 | FileCheck %s
// CHECK: 1:27: TypedefDecl=__darwin_size_t:1:27 (Definition) Extent=[1:1 - 1:42]
// CHECK: 2:25: TypedefDecl=size_t:2:25 (Definition) Extent=[2:1 - 2:31]
// CHECK: 2:9: TypeRef=__darwin_size_t:1:27 Extent=[2:9 - 2:24]
// CHECK: 3:11: Namespace=std:3:11 (Definition) Extent=[3:1 - 5:2]
// CHECK: 4:44: ClassTemplate=pair:4:44 (Definition) Extent=[4:3 - 4:64]
// CHECK: 4:20: TemplateTypeParameter=_T1:4:20 (Definition) Extent=[4:14 - 4:23]
// CHECK: 4:31: TemplateTypeParameter=_T2:4:31 (Definition) Extent=[4:25 - 4:34]
// CHECK: 4:55: FieldDecl=second:4:55 (Definition) Extent=[4:51 - 4:61]
// CHECK: 6:8: UnexposedDecl=:6:8 (Definition) Extent=[6:1 - 9:2]
// CHECK: 7:7: FunctionDecl=memcmp:7:7 Extent=[7:3 - 7:49]
// CHECK: 7:26: ParmDecl=:7:26 (Definition) Extent=[7:14 - 7:26]
// CHECK: 7:40: ParmDecl=:7:40 (Definition) Extent=[7:28 - 7:40]
// CHECK: 7:48: ParmDecl=:7:48 (Definition) Extent=[7:42 - 7:48]
// CHECK: 7:42: TypeRef=size_t:2:25 Extent=[7:42 - 7:48]
// CHECK: 8:10: FunctionDecl=strlen:8:10 Extent=[8:3 - 8:30]
// CHECK: 8:3: TypeRef=size_t:2:25 Extent=[8:3 - 8:9]
// CHECK: 8:29: ParmDecl=:8:29 (Definition) Extent=[8:17 - 8:29]
// CHECK: 10:17: Namespace=clang:10:17 (Definition) Extent=[10:1 - 35:2]
// CHECK: 11:9: ClassDecl=IdentifierInfo:11:9 Extent=[11:3 - 11:23]
// CHECK: 12:9: ClassDecl=AttributeList:12:9 (Definition) Extent=[12:3 - 34:4]
// CHECK: 13:10: EnumDecl=Kind:13:10 (Definition) Extent=[13:5 - 32:6]
// CHECK: 14:7: EnumConstantDecl=AT_IBAction:14:7 (Definition) Extent=[14:7 - 14:18]
// CHECK: 14:20: EnumConstantDecl=AT_IBOutlet:14:20 (Definition) Extent=[14:20 - 14:31]
// CHECK: 14:33: EnumConstantDecl=AT_IBOutletCollection:14:33 (Definition) Extent=[14:33 - 14:54]
// CHECK: 15:7: EnumConstantDecl=AT_address_space:15:7 (Definition) Extent=[15:7 - 15:23]
// CHECK: 15:25: EnumConstantDecl=AT_alias:15:25 (Definition) Extent=[15:25 - 15:33]
// CHECK: 15:35: EnumConstantDecl=AT_aligned:15:35 (Definition) Extent=[15:35 - 15:45]
// CHECK: 15:47: EnumConstantDecl=AT_always_inline:15:47 (Definition) Extent=[15:47 - 15:63]
// CHECK: 16:7: EnumConstantDecl=AT_analyzer_noreturn:16:7 (Definition) Extent=[16:7 - 16:27]
// CHECK: 16:29: EnumConstantDecl=AT_annotate:16:29 (Definition) Extent=[16:29 - 16:40]
// CHECK: 16:42: EnumConstantDecl=AT_base_check:16:42 (Definition) Extent=[16:42 - 16:55]
// CHECK: 16:57: EnumConstantDecl=AT_blocks:16:57 (Definition) Extent=[16:57 - 16:66]
// CHECK: 17:7: EnumConstantDecl=AT_carries_dependency:17:7 (Definition) Extent=[17:7 - 17:28]
// CHECK: 17:30: EnumConstantDecl=AT_cdecl:17:30 (Definition) Extent=[17:30 - 17:38]
// CHECK: 17:40: EnumConstantDecl=AT_cleanup:17:40 (Definition) Extent=[17:40 - 17:50]
// CHECK: 17:52: EnumConstantDecl=AT_const:17:52 (Definition) Extent=[17:52 - 17:60]
// CHECK: 17:62: EnumConstantDecl=AT_constructor:17:62 (Definition) Extent=[17:62 - 17:76]
// CHECK: 18:7: EnumConstantDecl=AT_deprecated:18:7 (Definition) Extent=[18:7 - 18:20]
// CHECK: 18:22: EnumConstantDecl=AT_destructor:18:22 (Definition) Extent=[18:22 - 18:35]
// CHECK: 18:37: EnumConstantDecl=AT_dllexport:18:37 (Definition) Extent=[18:37 - 18:49]
// CHECK: 18:51: EnumConstantDecl=AT_dllimport:18:51 (Definition) Extent=[18:51 - 18:63]
// CHECK: 19:7: EnumConstantDecl=AT_ext_vector_type:19:7 (Definition) Extent=[19:7 - 19:25]
// CHECK: 19:27: EnumConstantDecl=AT_fastcall:19:27 (Definition) Extent=[19:27 - 19:38]
// CHECK: 19:40: EnumConstantDecl=AT_final:19:40 (Definition) Extent=[19:40 - 19:48]
// CHECK: 19:50: EnumConstantDecl=AT_format:19:50 (Definition) Extent=[19:50 - 19:59]
// CHECK: 19:61: EnumConstantDecl=AT_format_arg:19:61 (Definition) Extent=[19:61 - 19:74]
// CHECK: 20:7: EnumConstantDecl=AT_gnu_inline:20:7 (Definition) Extent=[20:7 - 20:20]
// CHECK: 20:22: EnumConstantDecl=AT_hiding:20:22 (Definition) Extent=[20:22 - 20:31]
// CHECK: 20:33: EnumConstantDecl=AT_malloc:20:33 (Definition) Extent=[20:33 - 20:42]
// CHECK: 20:44: EnumConstantDecl=AT_mode:20:44 (Definition) Extent=[20:44 - 20:51]
// CHECK: 20:53: EnumConstantDecl=AT_naked:20:53 (Definition) Extent=[20:53 - 20:61]
// CHECK: 20:63: EnumConstantDecl=AT_nodebug:20:63 (Definition) Extent=[20:63 - 20:73]
// CHECK: 21:7: EnumConstantDecl=AT_noinline:21:7 (Definition) Extent=[21:7 - 21:18]
// CHECK: 21:20: EnumConstantDecl=AT_no_instrument_function:21:20 (Definition) Extent=[21:20 - 21:45]
// CHECK: 21:47: EnumConstantDecl=AT_nonnull:21:47 (Definition) Extent=[21:47 - 21:57]
// CHECK: 21:59: EnumConstantDecl=AT_noreturn:21:59 (Definition) Extent=[21:59 - 21:70]
// CHECK: 22:7: EnumConstantDecl=AT_nothrow:22:7 (Definition) Extent=[22:7 - 22:17]
// CHECK: 22:19: EnumConstantDecl=AT_nsobject:22:19 (Definition) Extent=[22:19 - 22:30]
// CHECK: 22:32: EnumConstantDecl=AT_objc_exception:22:32 (Definition) Extent=[22:32 - 22:49]
// CHECK: 22:51: EnumConstantDecl=AT_override:22:51 (Definition) Extent=[22:51 - 22:62]
// CHECK: 23:7: EnumConstantDecl=AT_cf_returns_not_retained:23:7 (Definition) Extent=[23:7 - 23:33]
// CHECK: 23:35: EnumConstantDecl=AT_cf_returns_retained:23:35 (Definition) Extent=[23:35 - 23:57]
// CHECK: 24:7: EnumConstantDecl=AT_ns_returns_not_retained:24:7 (Definition) Extent=[24:7 - 24:33]
// CHECK: 24:35: EnumConstantDecl=AT_ns_returns_retained:24:35 (Definition) Extent=[24:35 - 24:57]
// CHECK: 24:59: EnumConstantDecl=AT_objc_gc:24:59 (Definition) Extent=[24:59 - 24:69]
// CHECK: 25:7: EnumConstantDecl=AT_overloadable:25:7 (Definition) Extent=[25:7 - 25:22]
// CHECK: 25:24: EnumConstantDecl=AT_ownership_holds:25:24 (Definition) Extent=[25:24 - 25:42]
// CHECK: 25:44: EnumConstantDecl=AT_ownership_returns:25:44 (Definition) Extent=[25:44 - 25:64]
// CHECK: 26:7: EnumConstantDecl=AT_ownership_takes:26:7 (Definition) Extent=[26:7 - 26:25]
// CHECK: 26:27: EnumConstantDecl=AT_packed:26:27 (Definition) Extent=[26:27 - 26:36]
// CHECK: 26:38: EnumConstantDecl=AT_pascal:26:38 (Definition) Extent=[26:38 - 26:47]
// CHECK: 26:49: EnumConstantDecl=AT_pure:26:49 (Definition) Extent=[26:49 - 26:56]
// CHECK: 26:58: EnumConstantDecl=AT_regparm:26:58 (Definition) Extent=[26:58 - 26:68]
// CHECK: 27:7: EnumConstantDecl=AT_section:27:7 (Definition) Extent=[27:7 - 27:17]
// CHECK: 27:19: EnumConstantDecl=AT_sentinel:27:19 (Definition) Extent=[27:19 - 27:30]
// CHECK: 27:32: EnumConstantDecl=AT_stdcall:27:32 (Definition) Extent=[27:32 - 27:42]
// CHECK: 27:44: EnumConstantDecl=AT_thiscall:27:44 (Definition) Extent=[27:44 - 27:55]
// CHECK: 27:57: EnumConstantDecl=AT_transparent_union:27:57 (Definition) Extent=[27:57 - 27:77]
// CHECK: 28:7: EnumConstantDecl=AT_unavailable:28:7 (Definition) Extent=[28:7 - 28:21]
// CHECK: 28:23: EnumConstantDecl=AT_unused:28:23 (Definition) Extent=[28:23 - 28:32]
// CHECK: 28:34: EnumConstantDecl=AT_used:28:34 (Definition) Extent=[28:34 - 28:41]
// CHECK: 28:43: EnumConstantDecl=AT_vecreturn:28:43 (Definition) Extent=[28:43 - 28:55]
// CHECK: 28:57: EnumConstantDecl=AT_vector_size:28:57 (Definition) Extent=[28:57 - 28:71]
// CHECK: 29:7: EnumConstantDecl=AT_visibility:29:7 (Definition) Extent=[29:7 - 29:20]
// CHECK: 29:22: EnumConstantDecl=AT_warn_unused_result:29:22 (Definition) Extent=[29:22 - 29:43]
// CHECK: 29:45: EnumConstantDecl=AT_weak:29:45 (Definition) Extent=[29:45 - 29:52]
// CHECK: 29:54: EnumConstantDecl=AT_weakref:29:54 (Definition) Extent=[29:54 - 29:64]
// CHECK: 30:7: EnumConstantDecl=AT_weak_import:30:7 (Definition) Extent=[30:7 - 30:21]
// CHECK: 30:23: EnumConstantDecl=AT_reqd_wg_size:30:23 (Definition) Extent=[30:23 - 30:38]
// CHECK: 30:40: EnumConstantDecl=AT_init_priority:30:40 (Definition) Extent=[30:40 - 30:56]
// CHECK: 31:7:  EnumConstantDecl=AT_returns_twice:31:7 (Definition) Extent=[31:7 - 31:23]
// CHECK: 31:25: EnumConstantDecl=IgnoredAttribute:31:25 (Definition) Extent=[31:25 - 31:41]
// CHECK: 33:17: CXXMethod=getKind:33:17 (static) Extent=[33:5 - 33:53]
// CHECK: 33:12: TypeRef=enum clang::AttributeList::Kind:13:10 Extent=[33:12 - 33:16]
// CHECK: 33:48: ParmDecl=Name:33:48 (Definition) Extent=[33:25 - 33:52]
// CHECK: 33:31: TypeRef=class clang::IdentifierInfo:66:7 Extent=[33:31 - 33:45]
// CHECK: 36:8: FunctionDecl=magic_length:36:8 Extent=[36:1 - 36:35]
// CHECK: 36:1: TypeRef=size_t:2:25 Extent=[36:1 - 36:7]
// CHECK: 36:33: ParmDecl=s:36:33 (Definition) Extent=[36:21 - 36:34]
// CHECK: 37:11: Namespace=llvm:37:11 (Definition) Extent=[37:1 - 64:2]
// CHECK: 38:7: ClassDecl=StringRef:38:7 (Definition) Extent=[38:1 - 63:2]
// CHECK: 39:1: CXXAccessSpecifier=:39:1 (Definition) Extent=[39:1 - 39:8]
// CHECK: 40:23: TypedefDecl=iterator:40:23 (Definition) Extent=[40:3 - 40:31]
// CHECK: 41:23: VarDecl=npos:41:23 Extent=[41:3 - 41:40]
// CHECK: 41:16: TypeRef=size_t:2:25 Extent=[41:16 - 41:22]
// CHECK: 41:30: UnaryOperator= Extent=[41:30 - 41:40]
// CHECK: 41:31: CXXFunctionalCastExpr= Extent=[41:31 - 41:40]
// CHECK: 41:31: TypeRef=size_t:2:25 Extent=[41:31 - 41:37]
// CHECK: 41:38: UnexposedExpr= Extent=[41:38 - 41:39]
// CHECK: 41:38: IntegerLiteral= Extent=[41:38 - 41:39]
// CHECK: 42:1: CXXAccessSpecifier=:42:1 (Definition) Extent=[42:1 - 42:9]
// CHECK: 43:15: FieldDecl=Data:43:15 (Definition) Extent=[43:3 - 43:19]
// CHECK: 44:10: FieldDecl=Length:44:10 (Definition) Extent=[44:3 - 44:16]
// CHECK: 44:3: TypeRef=size_t:2:25 Extent=[44:3 - 44:9]
// CHECK: 45:17: CXXMethod=min:45:17 (Definition) (static) Extent=[45:3 - 45:66]
// CHECK: 45:10: TypeRef=size_t:2:25 Extent=[45:10 - 45:16]
// CHECK: 45:28: ParmDecl=a:45:28 (Definition) Extent=[45:21 - 45:29]
// CHECK: 45:21: TypeRef=size_t:2:25 Extent=[45:21 - 45:27]
// CHECK: 45:38: ParmDecl=b:45:38 (Definition) Extent=[45:31 - 45:39]
// CHECK: 45:31: TypeRef=size_t:2:25 Extent=[45:31 - 45:37]
// CHECK: 45:41: CompoundStmt= Extent=[45:41 - 45:66]
// CHECK: 45:43: ReturnStmt= Extent=[45:43 - 45:63]
// CHECK: 45:50: ConditionalOperator= Extent=[45:50 - 45:63]
// CHECK: 45:50: BinaryOperator= Extent=[45:50 - 45:55]
// CHECK: 45:50: DeclRefExpr=a:45:28 Extent=[45:50 - 45:51]
// CHECK: 45:54: DeclRefExpr=b:45:38 Extent=[45:54 - 45:55]
// CHECK: 45:58: DeclRefExpr=a:45:28 Extent=[45:58 - 45:59]
// CHECK: 45:62: DeclRefExpr=b:45:38 Extent=[45:62 - 45:63]
// CHECK: 46:1: CXXAccessSpecifier=:46:1 (Definition) Extent=[46:1 - 46:8]
// CHECK: 47:3: CXXConstructor=StringRef:47:3 (Definition) (default constructor) Extent=[47:3 - 47:37]
// CHECK: 47:16: MemberRef=Data:43:15 Extent=[47:16 - 47:20]
// CHECK: 47:21: UnexposedExpr= Extent=[47:21 - 47:22]
// CHECK: 47:21: IntegerLiteral= Extent=[47:21 - 47:22]
// CHECK: 47:25: MemberRef=Length:44:10 Extent=[47:25 - 47:31]
// CHECK: 47:32: UnexposedExpr= Extent=[47:32 - 47:33]
// CHECK: 47:32: IntegerLiteral= Extent=[47:32 - 47:33]
// CHECK: 47:35: CompoundStmt= Extent=[47:35 - 47:37]
// CHECK: 48:3: CXXConstructor=StringRef:48:3 (Definition) (converting constructor) Extent=[48:3 - 48:71]
// CHECK: 48:25: ParmDecl=Str:48:25 (Definition) Extent=[48:13 - 48:28]
// CHECK: 48:32: MemberRef=Data:43:15 Extent=[48:32 - 48:36]
// CHECK: 48:37: DeclRefExpr=Str:48:25 Extent=[48:37 - 48:40]
// CHECK: 48:43: MemberRef=Length:44:10 Extent=[48:43 - 48:49]
// CHECK: 48:50: CallExpr=magic_length:36:8 Extent=[48:50 - 48:67]
// CHECK: 48:50: UnexposedExpr=magic_length:36:8 Extent=[48:50 - 48:62]
// CHECK: 48:50: DeclRefExpr=magic_length:36:8 Extent=[48:50 - 48:62]
// CHECK: 48:63: DeclRefExpr=Str:48:25 Extent=[48:63 - 48:66]
// CHECK: 48:69: CompoundStmt= Extent=[48:69 - 48:71]
// CHECK: 49:3: CXXConstructor=StringRef:49:3 (Definition) Extent=[49:3 - 49:77]
// CHECK: 49:25: ParmDecl=data:49:25 (Definition) Extent=[49:13 - 49:29]
// CHECK: 49:38: ParmDecl=length:49:38 (Definition) Extent=[49:31 - 49:44]
// CHECK: 49:31: TypeRef=size_t:2:25 Extent=[49:31 - 49:37]
// CHECK: 49:48: MemberRef=Data:43:15 Extent=[49:48 - 49:52]
// CHECK: 49:53: DeclRefExpr=data:49:25 Extent=[49:53 - 49:57]
// CHECK: 49:60: MemberRef=Length:44:10 Extent=[49:60 - 49:66]
// CHECK: 49:67: DeclRefExpr=length:49:38 Extent=[49:67 - 49:73]
// CHECK: 49:75: CompoundStmt= Extent=[49:75 - 49:77]
// CHECK: 50:12: CXXMethod=end:50:12 (Definition) (const) Extent=[50:3 - 50:40] [access=public]
// CHECK: 50:3: TypeRef=iterator:40:23 Extent=[50:3 - 50:11]
// CHECK: 50:24: CompoundStmt= Extent=[50:24 - 50:40]
// CHECK: 50:26: ReturnStmt= Extent=[50:26 - 50:37]
// CHECK: 50:33: MemberRefExpr=Data:43:15 Extent=[50:33 - 50:37]
// CHECK: 51:10: CXXMethod=size:51:10 (Definition) (const) Extent=[51:3 - 51:41] [access=public]
// CHECK: 51:3: TypeRef=size_t:2:25 Extent=[51:3 - 51:9]
// CHECK: 51:23: CompoundStmt= Extent=[51:23 - 51:41]
// CHECK: 51:25: ReturnStmt= Extent=[51:25 - 51:38]
// CHECK: 51:32: MemberRefExpr=Length:44:10 Extent=[51:32 - 51:38]
// CHECK: 52:8: CXXMethod=startswith:52:8 (Definition) (const) Extent=[52:3 - 55:4] [access=public]
// CHECK: 52:29: ParmDecl=Prefix:52:29 (Definition) Extent=[52:19 - 52:35]
// CHECK: 52:19: TypeRef=class llvm::StringRef:38:7 Extent=[52:19 - 52:28]
// CHECK: 52:43: CompoundStmt= Extent=[52:43 - 55:4]
// CHECK: 53:5: ReturnStmt= Extent=[53:5 - 54:56]
// CHECK: 53:12: BinaryOperator= Extent=[53:12 - 54:56]
// CHECK: 53:12: BinaryOperator= Extent=[53:12 - 53:35]
// CHECK: 53:12: UnexposedExpr=Length:44:10 Extent=[53:12 - 53:18]
// CHECK: 53:12: MemberRefExpr=Length:44:10 Extent=[53:12 - 53:18]
// CHECK: 53:29: MemberRefExpr=Length:44:10 SingleRefName=[53:29 - 53:35] RefName=[53:29 - 53:35] Extent=[53:22 - 53:35]
// CHECK: 53:22: DeclRefExpr=Prefix:52:29 Extent=[53:22 - 53:28]
// CHECK: 54:11: BinaryOperator= Extent=[54:11 - 54:56]
// CHECK: 54:11: CallExpr=memcmp:7:7 Extent=[54:11 - 54:51]
// CHECK: 54:11: UnexposedExpr=memcmp:7:7 Extent=[54:11 - 54:17]
// CHECK: 54:11: DeclRefExpr=memcmp:7:7 Extent=[54:11 - 54:17]
// CHECK: 54:18: UnexposedExpr=Data:43:15 Extent=[54:18 - 54:22]
// CHECK: 54:18: MemberRefExpr=Data:43:15 Extent=[54:18 - 54:22]
// CHECK: 54:31: UnexposedExpr=Data:43:15 Extent=[54:24 - 54:35]
// CHECK: 54:31: MemberRefExpr=Data:43:15 SingleRefName=[54:31 - 54:35] RefName=[54:31 - 54:35] Extent=[54:24 - 54:35]
// CHECK: 54:24: DeclRefExpr=Prefix:52:29 Extent=[54:24 - 54:30]
// CHECK: 54:44: MemberRefExpr=Length:44:10 SingleRefName=[54:44 - 54:50] RefName=[54:44 - 54:50] Extent=[54:37 - 54:50]
// CHECK: 54:37: DeclRefExpr=Prefix:52:29 Extent=[54:37 - 54:43]
// CHECK: 54:55: IntegerLiteral= Extent=[54:55 - 54:56]
// CHECK: 56:8: CXXMethod=endswith:56:8 (Definition) (const) Extent=[56:3 - 59:4] [access=public]
// CHECK: 56:27: ParmDecl=Suffix:56:27 (Definition) Extent=[56:17 - 56:33]
// CHECK: 56:17: TypeRef=class llvm::StringRef:38:7 Extent=[56:17 - 56:26]
// CHECK: 56:41: CompoundStmt= Extent=[56:41 - 59:4]
// CHECK: 57:5: ReturnStmt= Extent=[57:5 - 58:69]
// CHECK: 57:12: BinaryOperator= Extent=[57:12 - 58:69]
// CHECK: 57:12: BinaryOperator= Extent=[57:12 - 57:35]
// CHECK: 57:12: UnexposedExpr=Length:44:10 Extent=[57:12 - 57:18]
// CHECK: 57:12: MemberRefExpr=Length:44:10 Extent=[57:12 - 57:18]
// CHECK: 57:29: MemberRefExpr=Length:44:10 SingleRefName=[57:29 - 57:35] RefName=[57:29 - 57:35] Extent=[57:22 - 57:35]
// CHECK: 57:22: DeclRefExpr=Suffix:56:27 Extent=[57:22 - 57:28]
// CHECK: 58:7: BinaryOperator= Extent=[58:7 - 58:69]
// CHECK: 58:7: CallExpr=memcmp:7:7 Extent=[58:7 - 58:64]
// CHECK: 58:7: UnexposedExpr=memcmp:7:7 Extent=[58:7 - 58:13]
// CHECK: 58:7: DeclRefExpr=memcmp:7:7 Extent=[58:7 - 58:13]
// CHECK: 58:14: UnexposedExpr= Extent=[58:14 - 58:35]
// CHECK: 58:14: BinaryOperator= Extent=[58:14 - 58:35]
// CHECK: 58:14: CallExpr=end:50:12 Extent=[58:14 - 58:19]
// CHECK: 58:14: MemberRefExpr=end:50:12 Extent=[58:14 - 58:17]
// CHECK: 58:29: MemberRefExpr=Length:44:10 SingleRefName=[58:29 - 58:35] RefName=[58:29 - 58:35] Extent=[58:22 - 58:35]
// CHECK: 58:22: DeclRefExpr=Suffix:56:27 Extent=[58:22 - 58:28]
// CHECK: 58:44: UnexposedExpr=Data:43:15 Extent=[58:37 - 58:48]
// CHECK: 58:44: MemberRefExpr=Data:43:15 SingleRefName=[58:44 - 58:48] RefName=[58:44 - 58:48] Extent=[58:37 - 58:48]
// CHECK: 58:37: DeclRefExpr=Suffix:56:27 Extent=[58:37 - 58:43]
// CHECK: 58:57: MemberRefExpr=Length:44:10 SingleRefName=[58:57 - 58:63] RefName=[58:57 - 58:63] Extent=[58:50 - 58:63]
// CHECK: 58:50: DeclRefExpr=Suffix:56:27 Extent=[58:50 - 58:56]
// CHECK: 58:68: IntegerLiteral= Extent=[58:68 - 58:69]
// CHECK: 60:13: CXXMethod=substr:60:13 (Definition) (const) Extent=[60:3 - 62:4] [access=public]
// CHECK: 60:3: TypeRef=class llvm::StringRef:38:7 Extent=[60:3 - 60:12]
// CHECK: 60:27: ParmDecl=Start:60:27 (Definition) Extent=[60:20 - 60:32]
// CHECK: 60:20: TypeRef=size_t:2:25 Extent=[60:20 - 60:26]
// CHECK: 60:41: ParmDecl=N:60:41 (Definition) Extent=[60:34 - 60:49]
// CHECK: 60:34: TypeRef=size_t:2:25 Extent=[60:34 - 60:40]
// CHECK: 60:45: DeclRefExpr=npos:41:23 Extent=[60:45 - 60:49]
// CHECK: 60:57: CompoundStmt= Extent=[60:57 - 62:4]
// CHECK: 61:5: ReturnStmt= Extent=[61:5 - 61:59]
// CHECK: 61:12: CallExpr= Extent=[61:12 - 61:59]
// CHECK: 61:12: UnexposedExpr=StringRef:49:3 Extent=[61:12 - 61:59]
// CHECK: 61:12: CallExpr=StringRef:49:3 Extent=[61:12 - 61:59]
// CHECK: 61:12: TypeRef=class llvm::StringRef:38:7 Extent=[61:12 - 61:21]
// CHECK: 61:22: BinaryOperator= Extent=[61:22 - 61:34]
// CHECK: 61:22: UnexposedExpr=Data:43:15 Extent=[61:22 - 61:26]
// CHECK: 61:22: MemberRefExpr=Data:43:15 Extent=[61:22 - 61:26]
// CHECK: 61:29: DeclRefExpr=Start:60:27 Extent=[61:29 - 61:34]
// CHECK: 61:36: CallExpr=min:45:17 Extent=[61:36 - 61:58]
// CHECK: 61:36: UnexposedExpr=min:45:17 Extent=[61:36 - 61:39]
// CHECK: 61:36: DeclRefExpr=min:45:17 Extent=[61:36 - 61:39]
// CHECK: 61:40: DeclRefExpr=N:60:41 Extent=[61:40 - 61:41]
// CHECK: 61:43: BinaryOperator= Extent=[61:43 - 61:57]
// CHECK: 61:43: UnexposedExpr=Length:44:10 Extent=[61:43 - 61:49]
// CHECK: 61:43: MemberRefExpr=Length:44:10 Extent=[61:43 - 61:49]
// CHECK: 61:52: DeclRefExpr=Start:60:27 Extent=[61:52 - 61:57]
// CHECK: 65:11: Namespace=clang:65:11 (Definition) Extent=[65:1 - 81:2]
// CHECK: 66:7: ClassDecl=IdentifierInfo:66:7 (Definition) Extent=[66:1 - 80:2]
// CHECK: 67:1: CXXAccessSpecifier=:67:1 (Definition) Extent=[67:1 - 67:8]
// CHECK: 67:8: CXXConstructor=IdentifierInfo:67:8 (default constructor) Extent=[67:8 - 67:24]
// CHECK: 68:15: CXXMethod=getNameStart:68:15 (Definition) (const) Extent=[68:3 - 71:4] [access=public]
// CHECK: 68:36: CompoundStmt= Extent=[68:36 - 71:4]
// CHECK: 69:5: DeclStmt= Extent=[69:5 - 69:65]
// CHECK: 69:54: TypedefDecl=actualtype:69:54 (Definition) Extent=[69:5 - 69:64]
// CHECK: 69:18: TemplateRef=pair:4:44 Extent=[69:18 - 69:22]
// CHECK: 69:25: TypeRef=class clang::IdentifierInfo:66:7 Extent=[69:25 - 69:39]
// CHECK: 70:5: ReturnStmt= Extent=[70:5 - 70:47]
// CHECK: 70:41: MemberRefExpr=second:4:55 SingleRefName=[70:41 - 70:47] RefName=[70:41 - 70:47] Extent=[70:12 - 70:47]
// CHECK: 70:12: ParenExpr= Extent=[70:12 - 70:39]
// CHECK: 70:13: CStyleCastExpr= Extent=[70:13 - 70:38]
// CHECK: 70:20: TypeRef=actualtype:69:54 Extent=[70:20 - 70:30]
// CHECK: 70:34: CXXThisExpr= Extent=[70:34 - 70:38]
// CHECK: 72:12: CXXMethod=getLength:72:12 (Definition) (const) Extent=[72:3 - 76:4] [access=public]
// CHECK: 72:30: CompoundStmt= Extent=[72:30 - 76:4]
// CHECK: 73:5: DeclStmt= Extent=[73:5 - 73:65]
// CHECK: 73:54: TypedefDecl=actualtype:73:54 (Definition) Extent=[73:5 - 73:64]
// CHECK: 73:18: TemplateRef=pair:4:44 Extent=[73:18 - 73:22]
// CHECK: 73:25: TypeRef=class clang::IdentifierInfo:66:7 Extent=[73:25 - 73:39]
// CHECK: 74:5: DeclStmt= Extent=[74:5 - 74:61]
// CHECK: 74:17: VarDecl=p:74:17 (Definition) Extent=[74:5 - 74:60]
// CHECK: 74:21: BinaryOperator= Extent=[74:21 - 74:60]
// CHECK: 74:50: UnexposedExpr=second:4:55 Extent=[74:21 - 74:56]
// CHECK: 74:50: MemberRefExpr=second:4:55 SingleRefName=[74:50 - 74:56] RefName=[74:50 - 74:56] Extent=[74:21 - 74:56]
// CHECK: 74:21: ParenExpr= Extent=[74:21 - 74:48]
// CHECK: 74:22: CStyleCastExpr= Extent=[74:22 - 74:47]
// CHECK: 74:29: TypeRef=actualtype:73:54 Extent=[74:29 - 74:39]
// CHECK: 74:43: CXXThisExpr= Extent=[74:43 - 74:47]
// CHECK: 74:59: IntegerLiteral= Extent=[74:59 - 74:60]
// CHECK: 75:5: ReturnStmt= Extent=[75:5 - 75:62]
// CHECK: 75:12: BinaryOperator= Extent=[75:12 - 75:62]
// CHECK: 75:12: ParenExpr= Extent=[75:12 - 75:58]
// CHECK: 75:13: BinaryOperator= Extent=[75:13 - 75:57]
// CHECK: 75:13: ParenExpr= Extent=[75:13 - 75:30]
// CHECK: 75:14: CStyleCastExpr= Extent=[75:14 - 75:29]
// CHECK: 75:25: UnexposedExpr= Extent=[75:25 - 75:29]
// CHECK: 75:25: UnexposedExpr= Extent=[75:25 - 75:29]
// CHECK: 75:25: ArraySubscriptExpr= Extent=[75:25 - 75:29]
// CHECK: 75:25: DeclRefExpr=p:74:17 Extent=[75:25 - 75:26]
// CHECK: 75:27: IntegerLiteral= Extent=[75:27 - 75:28]
// CHECK: 75:33: ParenExpr= Extent=[75:33 - 75:57]
// CHECK: 75:34: BinaryOperator= Extent=[75:34 - 75:56]
// CHECK: 75:34: ParenExpr= Extent=[75:34 - 75:51]
// CHECK: 75:35: CStyleCastExpr= Extent=[75:35 - 75:50]
// CHECK: 75:46: UnexposedExpr= Extent=[75:46 - 75:50]
// CHECK: 75:46: UnexposedExpr= Extent=[75:46 - 75:50]
// CHECK: 75:46: ArraySubscriptExpr= Extent=[75:46 - 75:50]
// CHECK: 75:46: DeclRefExpr=p:74:17 Extent=[75:46 - 75:47]
// CHECK: 75:48: IntegerLiteral= Extent=[75:48 - 75:49]
// CHECK: 75:55: IntegerLiteral= Extent=[75:55 - 75:56]
// CHECK: 75:61: UnexposedExpr= Extent=[75:61 - 75:62]
// CHECK: 75:61: IntegerLiteral= Extent=[75:61 - 75:62]
// CHECK: 77:19: CXXMethod=getName:77:19 (Definition) (const) Extent=[77:3 - 79:4] [access=public]
// CHECK: 77:35: CompoundStmt= Extent=[77:35 - 79:4]
// CHECK: 78:5: ReturnStmt= Extent=[78:5 - 78:56]
// CHECK: 78:12: CallExpr= Extent=[78:12 - 78:56]
// CHECK: 78:12: UnexposedExpr=StringRef:49:3 Extent=[78:12 - 78:56]
// CHECK: 78:12: CallExpr=StringRef:49:3 Extent=[78:12 - 78:56]
// CHECK: 78:28: CallExpr=getNameStart:68:15 Extent=[78:28 - 78:42]
// CHECK: 78:28: MemberRefExpr=getNameStart:68:15 Extent=[78:28 - 78:40]
// CHECK: 78:44: UnexposedExpr=getLength:72:12 Extent=[78:44 - 78:55]
// CHECK: 78:44: CallExpr=getLength:72:12 Extent=[78:44 - 78:55]
// CHECK: 78:44: MemberRefExpr=getLength:72:12 Extent=[78:44 - 78:53]
// CHECK: 82:11: Namespace=llvm:82:11 (Definition) Extent=[82:1 - 96:2]
// CHECK: 83:47: ClassTemplate=StringSwitch:83:47 (Definition) Extent=[83:1 - 95:2]
// CHECK: 83:21: TemplateTypeParameter=T:83:21 (Definition) Extent=[83:12 - 83:22]
// CHECK: 83:33: TemplateTypeParameter=R:83:33 (Definition) Extent=[83:24 - 83:38]
// CHECK: 84:13: FieldDecl=Str:84:13 (Definition) Extent=[84:3 - 84:16]
// CHECK: 84:3: TypeRef=class llvm::StringRef:38:7 Extent=[84:3 - 84:12]
// CHECK: 85:12: FieldDecl=Result:85:12 (Definition) Extent=[85:3 - 85:18]
// CHECK: 86:1: CXXAccessSpecifier=:86:1 (Definition) Extent=[86:1 - 86:8]
// CHECK: 87:12: CXXConstructor=StringSwitch<T, R>:87:12 (Definition) Extent=[87:3 - 87:64]
// CHECK: 87:35: ParmDecl=Str:87:35 (Definition) Extent=[87:25 - 87:38]
// CHECK: 87:25: TypeRef=class llvm::StringRef:38:7 Extent=[87:25 - 87:34]
// CHECK: 87:42: MemberRef=Str:84:13 Extent=[87:42 - 87:45]
// CHECK: 87:42: CallExpr=StringRef:38:7 Extent=[87:42 - 87:50]
// CHECK: 87:46: DeclRefExpr=Str:87:35 Extent=[87:46 - 87:49]
// CHECK: 87:52: MemberRef=Result:85:12 Extent=[87:52 - 87:58]
// CHECK: 87:58: UnexposedExpr= Extent=[87:58 - 87:61]
// CHECK: 87:59: IntegerLiteral= Extent=[87:59 - 87:60]
// CHECK: 87:62: CompoundStmt= Extent=[87:62 - 87:64]
// CHECK: 88:42: FunctionTemplate=Case:88:42 (Definition) Extent=[88:3 - 91:4]
// CHECK: 88:23: NonTypeTemplateParameter=N:88:23 (Definition) Extent=[88:14 - 88:24]
// CHECK: 88:60: ParmDecl=S:88:60 (Definition) Extent=[88:47 - 88:65]
// CHECK: 88:63: DeclRefExpr=N:88:23 Extent=[88:63 - 88:64]
// CHECK: 89:57: ParmDecl=Value:89:57 (Definition) Extent=[89:47 - 89:62]
// CHECK: 89:64: CompoundStmt= Extent=[89:64 - 91:4]
// CHECK: 90:5: ReturnStmt= Extent=[90:5 - 90:17]
// CHECK: 90:12: UnaryOperator= Extent=[90:12 - 90:17]
// CHECK: 90:13: CXXThisExpr= Extent=[90:13 - 90:17]
// CHECK: 92:5: CXXMethod=Default:92:5 (Definition) (const) Extent=[92:3 - 94:4] [access=public]
// CHECK: 92:23: ParmDecl=Value:92:23 (Definition) Extent=[92:13 - 92:28]
// CHECK: 92:36: CompoundStmt= Extent=[92:36 - 94:4]
// CHECK: 93:5: ReturnStmt= Extent=[93:5 - 93:17]
// CHECK: 93:12: DeclRefExpr=Value:92:23 Extent=[93:12 - 93:17]
// CHECK: 98:17: UsingDirective=:98:17 Extent=[98:1 - 98:22]
// CHECK: 98:17: NamespaceRef=clang:10:17 Extent=[98:17 - 98:22]
// CHECK: 100:36: CXXMethod=getKind:100:36 (Definition) (static) Extent=[100:1 - 186:2]
// CHECK: 100:21: TypeRef=class clang::AttributeList:12:9 Extent=[100:21 - 100:34]
// CHECK: 100:67: ParmDecl=Name:100:67 (Definition) Extent=[100:44 - 100:71]
// CHECK: 100:50: TypeRef=class clang::IdentifierInfo:66:7 Extent=[100:50 - 100:64]
// CHECK: 100:73: CompoundStmt= Extent=[100:73 - 186:2]
// CHECK: 101:3: DeclStmt= Extent=[101:3 - 101:46]
// CHECK: 101:19: VarDecl=AttrName:101:19 (Definition) Extent=[101:3 - 101:45]
// CHECK: 101:30: CallExpr= Extent=[101:30 - 101:45]
// CHECK: 101:30: UnexposedExpr=getName:77:19 Extent=[101:30 - 101:45]
// CHECK: 101:30: CallExpr=getName:77:19 Extent=[101:30 - 101:45]
// CHECK: 101:36: MemberRefExpr=getName:77:19 SingleRefName=[101:36 - 101:43] RefName=[101:36 - 101:43] Extent=[101:30 - 101:43]
// CHECK: 101:30: DeclRefExpr=Name:100:67 Extent=[101:30 - 101:34]
// CHECK: 102:3: IfStmt= Extent=[102:3 - 103:55]
// CHECK: 102:7: BinaryOperator= Extent=[102:7 - 102:59]
// CHECK: 102:7: CallExpr=startswith:52:8 Extent=[102:7 - 102:32]
// CHECK: 102:16: MemberRefExpr=startswith:52:8 SingleRefName=[102:16 - 102:26] RefName=[102:16 - 102:26] Extent=[102:7 - 102:26]
// CHECK: 102:7: UnexposedExpr=AttrName:101:19 Extent=[102:7 - 102:15]
// CHECK: 102:7: DeclRefExpr=AttrName:101:19 Extent=[102:7 - 102:15]
// CHECK: 102:27: CallExpr= Extent=[102:27 - 102:31]
// CHECK: 102:27: UnexposedExpr=StringRef:48:3 Extent=[102:27 - 102:31]
// CHECK: 102:27: UnexposedExpr=StringRef:48:3 Extent=[102:27 - 102:31]
// CHECK: 102:27: CallExpr=StringRef:48:3 Extent=[102:27 - 102:31]
// CHECK: 102:27: UnexposedExpr= Extent=[102:27 - 102:31]
// CHECK: 102:27: StringLiteral="__" Extent=[102:27 - 102:31]
// CHECK: 102:36: CallExpr=endswith:56:8 Extent=[102:36 - 102:59]
// CHECK: 102:45: MemberRefExpr=endswith:56:8 SingleRefName=[102:45 - 102:53] RefName=[102:45 - 102:53] Extent=[102:36 - 102:53]
// CHECK: 102:36: UnexposedExpr=AttrName:101:19 Extent=[102:36 - 102:44]
// CHECK: 102:36: DeclRefExpr=AttrName:101:19 Extent=[102:36 - 102:44]
// CHECK: 102:54: CallExpr= Extent=[102:54 - 102:58]
// CHECK: 102:54: UnexposedExpr=StringRef:48:3 Extent=[102:54 - 102:58]
// CHECK: 102:54: UnexposedExpr=StringRef:48:3 Extent=[102:54 - 102:58]
// CHECK: 102:54: CallExpr=StringRef:48:3 Extent=[102:54 - 102:58]
// CHECK: 102:54: UnexposedExpr= Extent=[102:54 - 102:58]
// CHECK: 102:54: StringLiteral="__" Extent=[102:54 - 102:58]
// CHECK: 103:5: CallExpr=operator=:38:7 Extent=[103:5 - 103:55]
// CHECK: 103:5: DeclRefExpr=AttrName:101:19 Extent=[103:5 - 103:13]
// CHECK: 103:14: UnexposedExpr=operator=:38:7
// CHECK: 103:14: DeclRefExpr=operator=:38:7
// CHECK: 103:16: UnexposedExpr=substr:60:13 Extent=[103:16 - 103:55]
// CHECK: 103:16: CallExpr=substr:60:13 Extent=[103:16 - 103:55]
// CHECK: 103:25: MemberRefExpr=substr:60:13 SingleRefName=[103:25 - 103:31] RefName=[103:25 - 103:31] Extent=[103:16 - 103:31]
// CHECK: 103:16: UnexposedExpr=AttrName:101:19 Extent=[103:16 - 103:24]
// CHECK: 103:16: DeclRefExpr=AttrName:101:19 Extent=[103:16 - 103:24]
// CHECK: 103:32: UnexposedExpr= Extent=[103:32 - 103:33]
// CHECK: 103:32: IntegerLiteral= Extent=[103:32 - 103:33]
// CHECK: 103:35: BinaryOperator= Extent=[103:35 - 103:54]
// CHECK: 103:35: CallExpr=size:51:10 Extent=[103:35 - 103:50]
// CHECK: 103:44: MemberRefExpr=size:51:10 SingleRefName=[103:44 - 103:48] RefName=[103:44 - 103:48] Extent=[103:35 - 103:48]
// CHECK: 103:35: UnexposedExpr=AttrName:101:19 Extent=[103:35 - 103:43]
// CHECK: 103:35: DeclRefExpr=AttrName:101:19 Extent=[103:35 - 103:43]
// CHECK: 103:53: UnexposedExpr= Extent=[103:53 - 103:54]
// CHECK: 103:53: IntegerLiteral= Extent=[103:53 - 103:54]
// CHECK: 105:3: ReturnStmt= Extent=[105:3 - 185:31]
// CHECK: 105:10: CallExpr=Default:92:5 Extent=[105:10 - 185:31]
// CHECK: 185:6: MemberRefExpr=Default:92:5 SingleRefName=[185:6 - 185:13] RefName=[185:6 - 185:13] Extent=[105:10 - 185:13]
// CHECK: 105:10: UnexposedExpr=Case:88:42 Extent=[105:10 - 184:33]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 184:33]
// CHECK: 184:6: MemberRefExpr=Case:88:42 SingleRefName=[184:6 - 184:10] RefName=[184:6 - 184:10] Extent=[105:10 - 184:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 183:37]
// CHECK: 183:6: MemberRefExpr=Case:88:42 SingleRefName=[183:6 - 183:10] RefName=[183:6 - 183:10] Extent=[105:10 - 183:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 182:37]
// CHECK: 182:6: MemberRefExpr=Case:88:42 SingleRefName=[182:6 - 182:10] RefName=[182:6 - 182:10] Extent=[105:10 - 182:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 181:35]
// CHECK: 181:6: MemberRefExpr=Case:88:42 SingleRefName=[181:6 - 181:10] RefName=[181:6 - 181:10] Extent=[105:10 - 181:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 180:31]
// CHECK: 180:6: MemberRefExpr=Case:88:42 SingleRefName=[180:6 - 180:10] RefName=[180:6 - 180:10] Extent=[105:10 - 180:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 179:31]
// CHECK: 179:6: MemberRefExpr=Case:88:42 SingleRefName=[179:6 - 179:10] RefName=[179:6 - 179:10] Extent=[105:10 - 179:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 178:35]
// CHECK: 178:6: MemberRefExpr=Case:88:42 SingleRefName=[178:6 - 178:10] RefName=[178:6 - 178:10] Extent=[105:10 - 178:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 177:63]
// CHECK: 177:6: MemberRefExpr=Case:88:42 SingleRefName=[177:6 - 177:10] RefName=[177:6 - 177:10] Extent=[105:10 - 177:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 176:45]
// CHECK: 176:6: MemberRefExpr=Case:88:42 SingleRefName=[176:6 - 176:10] RefName=[176:6 - 176:10] Extent=[105:10 - 176:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 175:51]
// CHECK: 175:6: MemberRefExpr=Case:88:42 SingleRefName=[175:6 - 175:10] RefName=[175:6 - 175:10] Extent=[105:10 - 175:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 174:49]
// CHECK: 174:6: MemberRefExpr=Case:88:42 SingleRefName=[174:6 - 174:10] RefName=[174:6 - 174:10] Extent=[105:10 - 174:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 173:49]
// CHECK: 173:6: MemberRefExpr=Case:88:42 SingleRefName=[173:6 - 173:10] RefName=[173:6 - 173:10] Extent=[105:10 - 173:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 172:53]
// CHECK: 172:6: MemberRefExpr=Case:88:42 SingleRefName=[172:6 - 172:10] RefName=[172:6 - 172:10] Extent=[105:10 - 172:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 171:57]
// CHECK: 171:6: MemberRefExpr=Case:88:42 SingleRefName=[171:6 - 171:10] RefName=[171:6 - 171:10] Extent=[105:10 - 171:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 170:65]
// CHECK: 170:6: MemberRefExpr=Case:88:42 SingleRefName=[170:6 - 170:10] RefName=[170:6 - 170:10] Extent=[105:10 - 170:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 169:57]
// CHECK: 169:6: MemberRefExpr=Case:88:42 SingleRefName=[169:6 - 169:10] RefName=[169:6 - 169:10] Extent=[105:10 - 169:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 168:65]
// CHECK: 168:6: MemberRefExpr=Case:88:42 SingleRefName=[168:6 - 168:10] RefName=[168:6 - 168:10] Extent=[105:10 - 168:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 167:55]
// CHECK: 167:6: MemberRefExpr=Case:88:42 SingleRefName=[167:6 - 167:10] RefName=[167:6 - 167:10] Extent=[105:10 - 167:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 166:55]
// CHECK: 166:6: MemberRefExpr=Case:88:42 SingleRefName=[166:6 - 166:10] RefName=[166:6 - 166:10] Extent=[105:10 - 166:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 165:53]
// CHECK: 165:6: MemberRefExpr=Case:88:42 SingleRefName=[165:6 - 165:10] RefName=[165:6 - 165:10] Extent=[105:10 - 165:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 164:53]
// CHECK: 164:6: MemberRefExpr=Case:88:42 SingleRefName=[164:6 - 164:10] RefName=[164:6 - 164:10] Extent=[105:10 - 164:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 163:49]
// CHECK: 163:6: MemberRefExpr=Case:88:42 SingleRefName=[163:6 - 163:10] RefName=[163:6 - 163:10] Extent=[105:10 - 163:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 162:47]
// CHECK: 162:6: MemberRefExpr=Case:88:42 SingleRefName=[162:6 - 162:10] RefName=[162:6 - 162:10] Extent=[105:10 - 162:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 161:45]
// CHECK: 161:6: MemberRefExpr=Case:88:42 SingleRefName=[161:6 - 161:10] RefName=[161:6 - 161:10] Extent=[105:10 - 161:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 160:45]
// CHECK: 160:6: MemberRefExpr=Case:88:42 SingleRefName=[160:6 - 160:10] RefName=[160:6 - 160:10] Extent=[105:10 - 160:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 159:45]
// CHECK: 159:6: MemberRefExpr=Case:88:42 SingleRefName=[159:6 - 159:10] RefName=[159:6 - 159:10] Extent=[105:10 - 159:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 158:45]
// CHECK: 158:6: MemberRefExpr=Case:88:42 SingleRefName=[158:6 - 158:10] RefName=[158:6 - 158:10] Extent=[105:10 - 158:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 157:43]
// CHECK: 157:6: MemberRefExpr=Case:88:42 SingleRefName=[157:6 - 157:10] RefName=[157:6 - 157:10] Extent=[105:10 - 157:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 156:41]
// CHECK: 156:6: MemberRefExpr=Case:88:42 SingleRefName=[156:6 - 156:10] RefName=[156:6 - 156:10] Extent=[105:10 - 156:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 155:41]
// CHECK: 155:6: MemberRefExpr=Case:88:42 SingleRefName=[155:6 - 155:10] RefName=[155:6 - 155:10] Extent=[105:10 - 155:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 154:41]
// CHECK: 154:6: MemberRefExpr=Case:88:42 SingleRefName=[154:6 - 154:10] RefName=[154:6 - 154:10] Extent=[105:10 - 154:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 153:37]
// CHECK: 153:6: MemberRefExpr=Case:88:42 SingleRefName=[153:6 - 153:10] RefName=[153:6 - 153:10] Extent=[105:10 - 153:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 152:41]
// CHECK: 152:6: MemberRefExpr=Case:88:42 SingleRefName=[152:6 - 152:10] RefName=[152:6 - 152:10] Extent=[105:10 - 152:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 151:39]
// CHECK: 151:6: MemberRefExpr=Case:88:42 SingleRefName=[151:6 - 151:10] RefName=[151:6 - 151:10] Extent=[105:10 - 151:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 150:39]
// CHECK: 150:6: MemberRefExpr=Case:88:42 SingleRefName=[150:6 - 150:10] RefName=[150:6 - 150:10] Extent=[105:10 - 150:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 149:39]
// CHECK: 149:6: MemberRefExpr=Case:88:42 SingleRefName=[149:6 - 149:10] RefName=[149:6 - 149:10] Extent=[105:10 - 149:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 148:39]
// CHECK: 148:6: MemberRefExpr=Case:88:42 SingleRefName=[148:6 - 148:10] RefName=[148:6 - 148:10] Extent=[105:10 - 148:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 147:39]
// CHECK: 147:6: MemberRefExpr=Case:88:42 SingleRefName=[147:6 - 147:10] RefName=[147:6 - 147:10] Extent=[105:10 - 147:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 146:39]
// CHECK: 146:6: MemberRefExpr=Case:88:42 SingleRefName=[146:6 - 146:10] RefName=[146:6 - 146:10] Extent=[105:10 - 146:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 145:41]
// CHECK: 145:6: MemberRefExpr=Case:88:42 SingleRefName=[145:6 - 145:10] RefName=[145:6 - 145:10] Extent=[105:10 - 145:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 144:37]
// CHECK: 144:6: MemberRefExpr=Case:88:42 SingleRefName=[144:6 - 144:10] RefName=[144:6 - 144:10] Extent=[105:10 - 144:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 143:37]
// CHECK: 143:6: MemberRefExpr=Case:88:42 SingleRefName=[143:6 - 143:10] RefName=[143:6 - 143:10] Extent=[105:10 - 143:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 142:35]
// CHECK: 142:6: MemberRefExpr=Case:88:42 SingleRefName=[142:6 - 142:10] RefName=[142:6 - 142:10] Extent=[105:10 - 142:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 141:35]
// CHECK: 141:6: MemberRefExpr=Case:88:42 SingleRefName=[141:6 - 141:10] RefName=[141:6 - 141:10] Extent=[105:10 - 141:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 140:35]
// CHECK: 140:6: MemberRefExpr=Case:88:42 SingleRefName=[140:6 - 140:10] RefName=[140:6 - 140:10] Extent=[105:10 - 140:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 139:35]
// CHECK: 139:6: MemberRefExpr=Case:88:42 SingleRefName=[139:6 - 139:10] RefName=[139:6 - 139:10] Extent=[105:10 - 139:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 138:35]
// CHECK: 138:6: MemberRefExpr=Case:88:42 SingleRefName=[138:6 - 138:10] RefName=[138:6 - 138:10] Extent=[105:10 - 138:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 137:55]
// CHECK: 137:6: MemberRefExpr=Case:88:42 SingleRefName=[137:6 - 137:10] RefName=[137:6 - 137:10] Extent=[105:10 - 137:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 136:35]
// CHECK: 136:6: MemberRefExpr=Case:88:42 SingleRefName=[136:6 - 136:10] RefName=[136:6 - 136:10] Extent=[105:10 - 136:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 135:35]
// CHECK: 135:6: MemberRefExpr=Case:88:42 SingleRefName=[135:6 - 135:10] RefName=[135:6 - 135:10] Extent=[105:10 - 135:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 134:35]
// CHECK: 134:6: MemberRefExpr=Case:88:42 SingleRefName=[134:6 - 134:10] RefName=[134:6 - 134:10] Extent=[105:10 - 134:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 133:35]
// CHECK: 133:6: MemberRefExpr=Case:88:42 SingleRefName=[133:6 - 133:10] RefName=[133:6 - 133:10] Extent=[105:10 - 133:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 132:33]
// CHECK: 132:6: MemberRefExpr=Case:88:42 SingleRefName=[132:6 - 132:10] RefName=[132:6 - 132:10] Extent=[105:10 - 132:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 131:33]
// CHECK: 131:6: MemberRefExpr=Case:88:42 SingleRefName=[131:6 - 131:10] RefName=[131:6 - 131:10] Extent=[105:10 - 131:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 130:33]
// CHECK: 130:6: MemberRefExpr=Case:88:42 SingleRefName=[130:6 - 130:10] RefName=[130:6 - 130:10] Extent=[105:10 - 130:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 129:33]
// CHECK: 129:6: MemberRefExpr=Case:88:42 SingleRefName=[129:6 - 129:10] RefName=[129:6 - 129:10] Extent=[105:10 - 129:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 128:33]
// CHECK: 128:6: MemberRefExpr=Case:88:42 SingleRefName=[128:6 - 128:10] RefName=[128:6 - 128:10] Extent=[105:10 - 128:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 127:33]
// CHECK: 127:6: MemberRefExpr=Case:88:42 SingleRefName=[127:6 - 127:10] RefName=[127:6 - 127:10] Extent=[105:10 - 127:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 126:33]
// CHECK: 126:6: MemberRefExpr=Case:88:42 SingleRefName=[126:6 - 126:10] RefName=[126:6 - 126:10] Extent=[105:10 - 126:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 125:29]
// CHECK: 125:6: MemberRefExpr=Case:88:42 SingleRefName=[125:6 - 125:10] RefName=[125:6 - 125:10] Extent=[105:10 - 125:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 124:33]
// CHECK: 124:6: MemberRefExpr=Case:88:42 SingleRefName=[124:6 - 124:10] RefName=[124:6 - 124:10] Extent=[105:10 - 124:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 123:33]
// CHECK: 123:6: MemberRefExpr=Case:88:42 SingleRefName=[123:6 - 123:10] RefName=[123:6 - 123:10] Extent=[105:10 - 123:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 122:31]
// CHECK: 122:6: MemberRefExpr=Case:88:42 SingleRefName=[122:6 - 122:10] RefName=[122:6 - 122:10] Extent=[105:10 - 122:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 121:31]
// CHECK: 121:6: MemberRefExpr=Case:88:42 SingleRefName=[121:6 - 121:10] RefName=[121:6 - 121:10] Extent=[105:10 - 121:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 120:31]
// CHECK: 120:6: MemberRefExpr=Case:88:42 SingleRefName=[120:6 - 120:10] RefName=[120:6 - 120:10] Extent=[105:10 - 120:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 119:31]
// CHECK: 119:6: MemberRefExpr=Case:88:42 SingleRefName=[119:6 - 119:10] RefName=[119:6 - 119:10] Extent=[105:10 - 119:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 118:31]
// CHECK: 118:6: MemberRefExpr=Case:88:42 SingleRefName=[118:6 - 118:10] RefName=[118:6 - 118:10] Extent=[105:10 - 118:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 117:31]
// CHECK: 117:6: MemberRefExpr=Case:88:42 SingleRefName=[117:6 - 117:10] RefName=[117:6 - 117:10] Extent=[105:10 - 117:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 116:31]
// CHECK: 116:6: MemberRefExpr=Case:88:42 SingleRefName=[116:6 - 116:10] RefName=[116:6 - 116:10] Extent=[105:10 - 116:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 115:29]
// CHECK: 115:6: MemberRefExpr=Case:88:42 SingleRefName=[115:6 - 115:10] RefName=[115:6 - 115:10] Extent=[105:10 - 115:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 114:29]
// CHECK: 114:6: MemberRefExpr=Case:88:42 SingleRefName=[114:6 - 114:10] RefName=[114:6 - 114:10] Extent=[105:10 - 114:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 113:29]
// CHECK: 113:6: MemberRefExpr=Case:88:42 SingleRefName=[113:6 - 113:10] RefName=[113:6 - 113:10] Extent=[105:10 - 113:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 112:31]
// CHECK: 112:6: MemberRefExpr=Case:88:42 SingleRefName=[112:6 - 112:10] RefName=[112:6 - 112:10] Extent=[105:10 - 112:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 111:29]
// CHECK: 111:6: MemberRefExpr=Case:88:42 SingleRefName=[111:6 - 111:10] RefName=[111:6 - 111:10] Extent=[105:10 - 111:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 110:27]
// CHECK: 110:6: MemberRefExpr=Case:88:42 SingleRefName=[110:6 - 110:10] RefName=[110:6 - 110:10] Extent=[105:10 - 110:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 109:27]
// CHECK: 109:6: MemberRefExpr=Case:88:42 SingleRefName=[109:6 - 109:10] RefName=[109:6 - 109:10] Extent=[105:10 - 109:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 108:27]
// CHECK: 108:6: MemberRefExpr=Case:88:42 SingleRefName=[108:6 - 108:10] RefName=[108:6 - 108:10] Extent=[105:10 - 108:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 107:33]
// CHECK: 107:6: MemberRefExpr=Case:88:42 SingleRefName=[107:6 - 107:10] RefName=[107:6 - 107:10] Extent=[105:10 - 107:10]
// CHECK: 105:10: CallExpr=Case:88:42 Extent=[105:10 - 106:27]
// CHECK: 106:6: MemberRefExpr=Case:88:42 SingleRefName=[106:6 - 106:10] RefName=[106:6 - 106:10] Extent=[105:10 - 106:10]
// CHECK: 105:10: CXXFunctionalCastExpr= Extent=[105:10 - 105:63]
// CHECK: 105:16: TemplateRef=StringSwitch:83:47 Extent=[105:16 - 105:28]
// CHECK: 105:10: CallExpr=StringSwitch:87:12 Extent=[105:10 - 105:63]
// CHECK: 105:54: CallExpr=StringRef:38:7 Extent=[105:54 - 105:62]
// CHECK: 105:54: UnexposedExpr=AttrName:101:19 Extent=[105:54 - 105:62]
// CHECK: 105:54: DeclRefExpr=AttrName:101:19 Extent=[105:54 - 105:62]
// CHECK: 106:11: StringLiteral="weak" Extent=[106:11 - 106:17]
// CHECK: 106:19: DeclRefExpr=AT_weak:29:45 Extent=[106:19 - 106:26]
// CHECK: 107:11: StringLiteral="weakref" Extent=[107:11 - 107:20]
// CHECK: 107:22: DeclRefExpr=AT_weakref:29:54 Extent=[107:22 - 107:32]
// CHECK: 108:11: StringLiteral="pure" Extent=[108:11 - 108:17]
// CHECK: 108:19: DeclRefExpr=AT_pure:26:49 Extent=[108:19 - 108:26]
// CHECK: 109:11: StringLiteral="mode" Extent=[109:11 - 109:17]
// CHECK: 109:19: DeclRefExpr=AT_mode:20:44 Extent=[109:19 - 109:26]
// CHECK: 110:11: StringLiteral="used" Extent=[110:11 - 110:17]
// CHECK: 110:19: DeclRefExpr=AT_used:28:34 Extent=[110:19 - 110:26]
// CHECK: 111:11: StringLiteral="alias" Extent=[111:11 - 111:18]
// CHECK: 111:20: DeclRefExpr=AT_alias:15:25 Extent=[111:20 - 111:28]
// CHECK: 112:11: StringLiteral="align" Extent=[112:11 - 112:18]
// CHECK: 112:20: DeclRefExpr=AT_aligned:15:35 Extent=[112:20 - 112:30]
// CHECK: 113:11: StringLiteral="final" Extent=[113:11 - 113:18]
// CHECK: 113:20: DeclRefExpr=AT_final:19:40 Extent=[113:20 - 113:28]
// CHECK: 114:11: StringLiteral="cdecl" Extent=[114:11 - 114:18]
// CHECK: 114:20: DeclRefExpr=AT_cdecl:17:30 Extent=[114:20 - 114:28]
// CHECK: 115:11: StringLiteral="const" Extent=[115:11 - 115:18]
// CHECK: 115:20: DeclRefExpr=AT_const:17:52 Extent=[115:20 - 115:28]
// CHECK: 116:11: StringLiteral="__const" Extent=[116:11 - 116:20]
// CHECK: 116:22: DeclRefExpr=AT_const:17:52 Extent=[116:22 - 116:30]
// CHECK: 117:11: StringLiteral="blocks" Extent=[117:11 - 117:19]
// CHECK: 117:21: DeclRefExpr=AT_blocks:16:57 Extent=[117:21 - 117:30]
// CHECK: 118:11: StringLiteral="format" Extent=[118:11 - 118:19]
// CHECK: 118:21: DeclRefExpr=AT_format:19:50 Extent=[118:21 - 118:30]
// CHECK: 119:11: StringLiteral="hiding" Extent=[119:11 - 119:19]
// CHECK: 119:21: DeclRefExpr=AT_hiding:20:22 Extent=[119:21 - 119:30]
// CHECK: 120:11: StringLiteral="malloc" Extent=[120:11 - 120:19]
// CHECK: 120:21: DeclRefExpr=AT_malloc:20:33 Extent=[120:21 - 120:30]
// CHECK: 121:11: StringLiteral="packed" Extent=[121:11 - 121:19]
// CHECK: 121:21: DeclRefExpr=AT_packed:26:27 Extent=[121:21 - 121:30]
// CHECK: 122:11: StringLiteral="unused" Extent=[122:11 - 122:19]
// CHECK: 122:21: DeclRefExpr=AT_unused:28:23 Extent=[122:21 - 122:30]
// CHECK: 123:11: StringLiteral="aligned" Extent=[123:11 - 123:20]
// CHECK: 123:22: DeclRefExpr=AT_aligned:15:35 Extent=[123:22 - 123:32]
// CHECK: 124:11: StringLiteral="cleanup" Extent=[124:11 - 124:20]
// CHECK: 124:22: DeclRefExpr=AT_cleanup:17:40 Extent=[124:22 - 124:32]
// CHECK: 125:11: StringLiteral="naked" Extent=[125:11 - 125:18]
// CHECK: 125:20: DeclRefExpr=AT_naked:20:53 Extent=[125:20 - 125:28]
// CHECK: 126:11: StringLiteral="nodebug" Extent=[126:11 - 126:20]
// CHECK: 126:22: DeclRefExpr=AT_nodebug:20:63 Extent=[126:22 - 126:32]
// CHECK: 127:11: StringLiteral="nonnull" Extent=[127:11 - 127:20]
// CHECK: 127:22: DeclRefExpr=AT_nonnull:21:47 Extent=[127:22 - 127:32]
// CHECK: 128:11: StringLiteral="nothrow" Extent=[128:11 - 128:20]
// CHECK: 128:22: DeclRefExpr=AT_nothrow:22:7 Extent=[128:22 - 128:32]
// CHECK: 129:11: StringLiteral="objc_gc" Extent=[129:11 - 129:20]
// CHECK: 129:22: DeclRefExpr=AT_objc_gc:24:59 Extent=[129:22 - 129:32]
// CHECK: 130:11: StringLiteral="regparm" Extent=[130:11 - 130:20]
// CHECK: 130:22: DeclRefExpr=AT_regparm:26:58 Extent=[130:22 - 130:32]
// CHECK: 131:11: StringLiteral="section" Extent=[131:11 - 131:20]
// CHECK: 131:22: DeclRefExpr=AT_section:27:7 Extent=[131:22 - 131:32]
// CHECK: 132:11: StringLiteral="stdcall" Extent=[132:11 - 132:20]
// CHECK: 132:22: DeclRefExpr=AT_stdcall:27:32 Extent=[132:22 - 132:32]
// CHECK: 133:11: StringLiteral="annotate" Extent=[133:11 - 133:21]
// CHECK: 133:23: DeclRefExpr=AT_annotate:16:29 Extent=[133:23 - 133:34]
// CHECK: 134:11: StringLiteral="fastcall" Extent=[134:11 - 134:21]
// CHECK: 134:23: DeclRefExpr=AT_fastcall:19:27 Extent=[134:23 - 134:34]
// CHECK: 135:11: StringLiteral="ibaction" Extent=[135:11 - 135:21]
// CHECK: 135:23: DeclRefExpr=AT_IBAction:14:7 Extent=[135:23 - 135:34]
// CHECK: 136:11: StringLiteral="iboutlet" Extent=[136:11 - 136:21]
// CHECK: 136:23: DeclRefExpr=AT_IBOutlet:14:20 Extent=[136:23 - 136:34]
// CHECK: 137:11: StringLiteral="iboutletcollection" Extent=[137:11 - 137:31]
// CHECK: 137:33: DeclRefExpr=AT_IBOutletCollection:14:33 Extent=[137:33 - 137:54]
// CHECK: 138:11: StringLiteral="noreturn" Extent=[138:11 - 138:21]
// CHECK: 138:23: DeclRefExpr=AT_noreturn:21:59 Extent=[138:23 - 138:34]
// CHECK: 139:11: StringLiteral="noinline" Extent=[139:11 - 139:21]
// CHECK: 139:23: DeclRefExpr=AT_noinline:21:7 Extent=[139:23 - 139:34]
// CHECK: 140:11: StringLiteral="override" Extent=[140:11 - 140:21]
// CHECK: 140:23: DeclRefExpr=AT_override:22:51 Extent=[140:23 - 140:34]
// CHECK: 141:11: StringLiteral="sentinel" Extent=[141:11 - 141:21]
// CHECK: 141:23: DeclRefExpr=AT_sentinel:27:19 Extent=[141:23 - 141:34]
// CHECK: 142:11: StringLiteral="NSObject" Extent=[142:11 - 142:21]
// CHECK: 142:23: DeclRefExpr=AT_nsobject:22:19 Extent=[142:23 - 142:34]
// CHECK: 143:11: StringLiteral="dllimport" Extent=[143:11 - 143:22]
// CHECK: 143:24: DeclRefExpr=AT_dllimport:18:51 Extent=[143:24 - 143:36]
// CHECK: 144:11: StringLiteral="dllexport" Extent=[144:11 - 144:22]
// CHECK: 144:24: DeclRefExpr=AT_dllexport:18:37 Extent=[144:24 - 144:36]
// CHECK: 145:11: StringLiteral="may_alias" Extent=[145:11 - 145:22]
// CHECK: 146:11: StringLiteral="base_check" Extent=[146:11 - 146:23]
// CHECK: 146:25: DeclRefExpr=AT_base_check:16:42 Extent=[146:25 - 146:38]
// CHECK: 147:11: StringLiteral="deprecated" Extent=[147:11 - 147:23]
// CHECK: 147:25: DeclRefExpr=AT_deprecated:18:7 Extent=[147:25 - 147:38]
// CHECK: 148:11: StringLiteral="visibility" Extent=[148:11 - 148:23]
// CHECK: 148:25: DeclRefExpr=AT_visibility:29:7 Extent=[148:25 - 148:38]
// CHECK: 149:11: StringLiteral="destructor" Extent=[149:11 - 149:23]
// CHECK: 149:25: DeclRefExpr=AT_destructor:18:22 Extent=[149:25 - 149:38]
// CHECK: 150:11: StringLiteral="format_arg" Extent=[150:11 - 150:23]
// CHECK: 150:25: DeclRefExpr=AT_format_arg:19:61 Extent=[150:25 - 150:38]
// CHECK: 151:11: StringLiteral="gnu_inline" Extent=[151:11 - 151:23]
// CHECK: 151:25: DeclRefExpr=AT_gnu_inline:20:7 Extent=[151:25 - 151:38]
// CHECK: 152:11: StringLiteral="weak_import" Extent=[152:11 - 152:24]
// CHECK: 152:26: DeclRefExpr=AT_weak_import:30:7 Extent=[152:26 - 152:40]
// CHECK: 153:11: StringLiteral="vecreturn" Extent=[153:11 - 153:22]
// CHECK: 153:24: DeclRefExpr=AT_vecreturn:28:43 Extent=[153:24 - 153:36]
// CHECK: 154:11: StringLiteral="vector_size" Extent=[154:11 - 154:24]
// CHECK: 154:26: DeclRefExpr=AT_vector_size:28:57 Extent=[154:26 - 154:40]
// CHECK: 155:11: StringLiteral="constructor" Extent=[155:11 - 155:24]
// CHECK: 155:26: DeclRefExpr=AT_constructor:17:62 Extent=[155:26 - 155:40]
// CHECK: 156:11: StringLiteral="unavailable" Extent=[156:11 - 156:24]
// CHECK: 156:26: DeclRefExpr=AT_unavailable:28:7 Extent=[156:26 - 156:40]
// CHECK: 157:11: StringLiteral="overloadable" Extent=[157:11 - 157:25]
// CHECK: 157:27: DeclRefExpr=AT_overloadable:25:7 Extent=[157:27 - 157:42]
// CHECK: 158:11: StringLiteral="address_space" Extent=[158:11 - 158:26]
// CHECK: 158:28: DeclRefExpr=AT_address_space:15:7 Extent=[158:28 - 158:44]
// CHECK: 159:11: StringLiteral="always_inline" Extent=[159:11 - 159:26]
// CHECK: 159:28: DeclRefExpr=AT_always_inline:15:47 Extent=[159:28 - 159:44]
// CHECK: 160:11: StringLiteral="returns_twice" Extent=[160:11 - 160:26]
// CHECK: 161:11: StringLiteral="vec_type_hint" Extent=[161:11 - 161:26]
// CHECK: 162:11: StringLiteral="objc_exception" Extent=[162:11 - 162:27]
// CHECK: 162:29: DeclRefExpr=AT_objc_exception:22:32 Extent=[162:29 - 162:46]
// CHECK: 163:11: StringLiteral="ext_vector_type" Extent=[163:11 - 163:28]
// CHECK: 163:30: DeclRefExpr=AT_ext_vector_type:19:7 Extent=[163:30 - 163:48]
// CHECK: 164:11: StringLiteral="transparent_union" Extent=[164:11 - 164:30]
// CHECK: 164:32: DeclRefExpr=AT_transparent_union:27:57 Extent=[164:32 - 164:52]
// CHECK: 165:11: StringLiteral="analyzer_noreturn" Extent=[165:11 - 165:30]
// CHECK: 165:32: DeclRefExpr=AT_analyzer_noreturn:16:7 Extent=[165:32 - 165:52]
// CHECK: 166:11: StringLiteral="warn_unused_result" Extent=[166:11 - 166:31]
// CHECK: 166:33: DeclRefExpr=AT_warn_unused_result:29:22 Extent=[166:33 - 166:54]
// CHECK: 167:11: StringLiteral="carries_dependency" Extent=[167:11 - 167:31]
// CHECK: 167:33: DeclRefExpr=AT_carries_dependency:17:7 Extent=[167:33 - 167:54]
// CHECK: 168:11: StringLiteral="ns_returns_not_retained" Extent=[168:11 - 168:36]
// CHECK: 168:38: DeclRefExpr=AT_ns_returns_not_retained:24:7 Extent=[168:38 - 168:64]
// CHECK: 169:11: StringLiteral="ns_returns_retained" Extent=[169:11 - 169:32]
// CHECK: 169:34: DeclRefExpr=AT_ns_returns_retained:24:35 Extent=[169:34 - 169:56]
// CHECK: 170:11: StringLiteral="cf_returns_not_retained" Extent=[170:11 - 170:36]
// CHECK: 170:38: DeclRefExpr=AT_cf_returns_not_retained:23:7 Extent=[170:38 - 170:64]
// CHECK: 171:11: StringLiteral="cf_returns_retained" Extent=[171:11 - 171:32]
// CHECK: 171:34: DeclRefExpr=AT_cf_returns_retained:23:35 Extent=[171:34 - 171:56]
// CHECK: 172:11: StringLiteral="ownership_returns" Extent=[172:11 - 172:30]
// CHECK: 172:32: DeclRefExpr=AT_ownership_returns:25:44 Extent=[172:32 - 172:52]
// CHECK: 173:11: StringLiteral="ownership_holds" Extent=[173:11 - 173:28]
// CHECK: 173:30: DeclRefExpr=AT_ownership_holds:25:24 Extent=[173:30 - 173:48]
// CHECK: 174:11: StringLiteral="ownership_takes" Extent=[174:11 - 174:28]
// CHECK: 174:30: DeclRefExpr=AT_ownership_takes:26:7 Extent=[174:30 - 174:48]
// CHECK: 175:11: StringLiteral="reqd_work_group_size" Extent=[175:11 - 175:33]
// CHECK: 175:35: DeclRefExpr=AT_reqd_wg_size:30:23 Extent=[175:35 - 175:50]
// CHECK: 176:11: StringLiteral="init_priority" Extent=[176:11 - 176:26]
// CHECK: 176:28: DeclRefExpr=AT_init_priority:30:40 Extent=[176:28 - 176:44]
// CHECK: 177:11: StringLiteral="no_instrument_function" Extent=[177:11 - 177:35]
// CHECK: 177:37: DeclRefExpr=AT_no_instrument_function:21:20 Extent=[177:37 - 177:62]
// CHECK: 178:11: StringLiteral="thiscall" Extent=[178:11 - 178:21]
// CHECK: 178:23: DeclRefExpr=AT_thiscall:27:44 Extent=[178:23 - 178:34]
// CHECK: 179:11: StringLiteral="pascal" Extent=[179:11 - 179:19]
// CHECK: 179:21: DeclRefExpr=AT_pascal:26:38 Extent=[179:21 - 179:30]
// CHECK: 180:11: StringLiteral="__cdecl" Extent=[180:11 - 180:20]
// CHECK: 180:22: DeclRefExpr=AT_cdecl:17:30 Extent=[180:22 - 180:30]
// CHECK: 181:11: StringLiteral="__stdcall" Extent=[181:11 - 181:22]
// CHECK: 181:24: DeclRefExpr=AT_stdcall:27:32 Extent=[181:24 - 181:34]
// CHECK: 182:11: StringLiteral="__fastcall" Extent=[182:11 - 182:23]
// CHECK: 182:25: DeclRefExpr=AT_fastcall:19:27 Extent=[182:25 - 182:36]
// CHECK: 183:11: StringLiteral="__thiscall" Extent=[183:11 - 183:23]
// CHECK: 183:25: DeclRefExpr=AT_thiscall:27:44 Extent=[183:25 - 183:36]
// CHECK: 184:11: StringLiteral="__pascal" Extent=[184:11 - 184:21]
// CHECK: 184:23: DeclRefExpr=AT_pascal:26:38 Extent=[184:23 - 184:32]
