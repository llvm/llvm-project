// RUN: not %clang_cc1 -triple x86_64-pc-linux -std=c++11 -ast-dump -fcolor-diagnostics %s | FileCheck --strict-whitespace %s
// REQUIRES: ansi-escape-sequences

/// <a>Hello</a>
/// <br/>
int Test __attribute__((unused));

/// Comment
void TestAttributedStmt() {
  switch (1) {
  case 1:
    [[clang::fallthrough]];
  case 2:
    ;
  }
}

class __attribute__((lockable)) Mutex {
  /// A variable
  int var1;
  /// Another variable
  ///
  /// Like the other variable, but different
  int var2;
} mu1, mu2;
int TestExpr __attribute__((guarded_by(mu1)));

struct Invalid {
  __attribute__((noinline)) Invalid(error);
} Invalid;

//CHECK: {{^}}[[GREEN:.\[0;1;32m]]TranslationUnitDecl[[RESET:.\[0m]][[Yellow:.\[0;33m]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]> [[Yellow]]<invalid sloc>[[RESET]]{{$}}
//CHECK: {{^}}[[Blue:.\[0;34m]]|-[[RESET]][[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]> [[Yellow]]<invalid sloc>[[RESET]] implicit[[CYAN:.\[0;1;36m]] __int128_t[[RESET]] [[Green:.\[0;32m]]'__int128'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]> [[Yellow]]<invalid sloc>[[RESET]] implicit[[CYAN]] __uint128_t[[RESET]] [[Green]]'unsigned __int128'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]TypedefDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]<invalid sloc>[[RESET]]> [[Yellow]]<invalid sloc>[[RESET]] implicit[[CYAN]] __builtin_va_list[[RESET]] [[Green]]'__va_list_tag [1]'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]{{.*}}ast-dump-color.cpp:6:1[[RESET]], [[Yellow]]col:5[[RESET]]> [[Yellow]]col:5[[RESET]][[CYAN]] Test[[RESET]] [[Green]]'int'[[RESET]]
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[BLUE:.\[0;1;34m]]UnusedAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:25[[RESET]]> unused{{$}}
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[Blue]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]], [[Yellow]]line:5:8[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]|   `-[[RESET]][[Blue]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]], [[Yellow]]line:5:8[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]|     |-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:4:4[[RESET]]> Text=" "{{$}}
//CHECK: {{^}}[[Blue]]|     |-[[RESET]][[Blue]]HTMLStartTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:5[[RESET]], [[Yellow]]col:7[[RESET]]> Name="a"{{$}}
//CHECK: {{^}}[[Blue]]|     |-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]], [[Yellow]]col:12[[RESET]]> Text="Hello"{{$}}
//CHECK: {{^}}[[Blue]]|     |-[[RESET]][[Blue]]HTMLEndTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:13[[RESET]], [[Yellow]]col:16[[RESET]]> Name="a"{{$}}
//CHECK: {{^}}[[Blue]]|     |-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:5:4[[RESET]]> Text=" "{{$}}
//CHECK: {{^}}[[Blue]]|     `-[[RESET]][[Blue]]HTMLStartTagComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:5[[RESET]], [[Yellow]]col:8[[RESET]]> Name="br" SelfClosing{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]FunctionDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:9:1[[RESET]], [[Yellow]]line:16:1[[RESET]]> [[Yellow]]line:9:6[[RESET]][[CYAN]] TestAttributedStmt[[RESET]] [[Green]]'void ()'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[MAGENTA:.\[0;1;35m]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:27[[RESET]], [[Yellow]]line:16:1[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[MAGENTA]]SwitchStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:10:3[[RESET]], [[Yellow]]line:15:3[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |   |-[[RESET]][[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:10:11[[RESET]]> [[Green]]'int'[[RESET]][[Cyan:.\[0;36m]][[RESET]][[Cyan]][[RESET]][[CYAN]] 1[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |   `-[[RESET]][[MAGENTA]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:14[[RESET]], [[Yellow]]line:15:3[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     |-[[RESET]][[MAGENTA]]CaseStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:11:3[[RESET]], [[Yellow]]line:12:27[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     | |-[[RESET]][[MAGENTA]]ConstantExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:11:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] Int: 1[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |     | | `-[[RESET]][[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] 1[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |     | `-[[RESET]][[MAGENTA]]AttributedStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:12:5[[RESET]], [[Yellow]]col:27[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     |   |-[[RESET]][[BLUE]]FallThroughAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:7[[RESET]], [[Yellow]]col:14[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     |   `-[[RESET]][[MAGENTA]]NullStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:27[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     `-[[RESET]][[MAGENTA]]CaseStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:13:3[[RESET]], [[Yellow]]line:14:5[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |       |-[[RESET]][[MAGENTA]]ConstantExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:13:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] Int: 2[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |       | `-[[RESET]][[MAGENTA]]IntegerLiteral[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Green]]'int'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]][[CYAN]] 2[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |       `-[[RESET]][[MAGENTA]]NullStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:14:5[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[Blue]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:8:4[[RESET]], [[Yellow]]col:11[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]|   `-[[RESET]][[Blue]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:4[[RESET]], [[Yellow]]col:11[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]|     `-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:4[[RESET]], [[Yellow]]col:11[[RESET]]> Text=" Comment"{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:1[[RESET]], [[Yellow]]line:25:1[[RESET]]> [[Yellow]]line:18:33[[RESET]] class[[CYAN]] Mutex[[RESET]] definition{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[BLUE]]CapabilityAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:22[[RESET]]> capability "mutex"{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]col:33[[RESET]]> [[Yellow]]col:33[[RESET]] implicit class[[CYAN]] Mutex[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]FieldDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:20:3[[RESET]], [[Yellow]]col:7[[RESET]]> [[Yellow]]col:7[[RESET]][[CYAN]] var1[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[Blue]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:19:6[[RESET]], [[Yellow]]col:16[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |   `-[[RESET]][[Blue]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:16[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     `-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:16[[RESET]]> Text=" A variable"{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]FieldDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:24:3[[RESET]], [[Yellow]]col:7[[RESET]]> [[Yellow]]col:7[[RESET]][[CYAN]] var2[[RESET]] [[Green]]'int'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[Blue]]FullComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:21:6[[RESET]], [[Yellow]]line:23:44[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |   |-[[RESET]][[Blue]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:21:6[[RESET]], [[Yellow]]col:22[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |   | `-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:22[[RESET]]> Text=" Another variable"{{$}}
//CHECK: {{^}}[[Blue]]| |   `-[[RESET]][[Blue]]ParagraphComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:23:6[[RESET]], [[Yellow]]col:44[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |     `-[[RESET]][[Blue]]TextComment[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:6[[RESET]], [[Yellow]]col:44[[RESET]]> Text=" Like the other variable, but different"{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:33[[RESET]]> [[Yellow]]col:33[[RESET]] implicit used[[CYAN]] Mutex[[RESET]] [[Green]]'void () noexcept'[[RESET]] inline{{.*$}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[MAGENTA]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Yellow]]col:33[[RESET]] implicit constexpr[[CYAN]] Mutex[[RESET]] [[Green]]'void (const Mutex &)'[[RESET]] inline{{ .*$}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Yellow]]col:33[[RESET]] [[Green]]'const Mutex &'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Yellow]]col:33[[RESET]] implicit constexpr[[CYAN]] Mutex[[RESET]] [[Green]]'void (Mutex &&)'[[RESET]] inline{{ .*$}}
//CHECK: {{^}}[[Blue]]|   `-[[RESET]][[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:33[[RESET]]> [[Yellow]]col:33[[RESET]] [[Green]]'Mutex &&'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]line:25:3[[RESET]]> [[Yellow]]col:3[[RESET]] referenced[[CYAN]] mu1[[RESET]] [[Green]]'class Mutex':'Mutex'[[RESET]]
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[MAGENTA]]CXXConstructExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:3[[RESET]]> [[Green]]'class Mutex':'Mutex'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]] [[Green]]'void () noexcept'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:18:1[[RESET]], [[Yellow]]line:25:8[[RESET]]> [[Yellow]]col:8[[RESET]][[CYAN]] mu2[[RESET]] [[Green]]'class Mutex':'Mutex'[[RESET]]
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[MAGENTA]]CXXConstructExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Green]]'class Mutex':'Mutex'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]] [[Green]]'void () noexcept'[[RESET]]{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:26:1[[RESET]], [[Yellow]]col:5[[RESET]]> [[Yellow]]col:5[[RESET]][[CYAN]] TestExpr[[RESET]] [[Green]]'int'[[RESET]]
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[BLUE]]GuardedByAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:29[[RESET]], [[Yellow]]col:43[[RESET]]>{{$}}
//CHECK: {{^}}[[Blue]]|   `-[[RESET]][[MAGENTA]]DeclRefExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:40[[RESET]]> [[Green]]'class Mutex':'Mutex'[[RESET]][[Cyan]] lvalue[[RESET]][[Cyan]][[RESET]] [[GREEN]]Var[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]][[CYAN]] 'mu1'[[RESET]] [[Green]]'class Mutex':'Mutex'[[RESET]] non_odr_use_unevaluated{{$}}
//CHECK: {{^}}[[Blue]]|-[[RESET]][[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:28:1[[RESET]], [[Yellow]]line:30:1[[RESET]]> [[Yellow]]line:28:8[[RESET]] struct[[CYAN]] Invalid[[RESET]] definition
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXRecordDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]col:8[[RESET]]> [[Yellow]]col:8[[RESET]] implicit referenced struct[[CYAN]] Invalid[[RESET]]
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:29:3[[RESET]], [[Yellow]]col:42[[RESET]]> [[Yellow]]col:29[[RESET]] invalid[[CYAN]] Invalid[[RESET]] [[Green]]'void (int)'[[RESET]]
//CHECK: {{^}}[[Blue]]| | |-[[RESET]][[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:37[[RESET]], [[Yellow]]<invalid sloc>[[RESET]]> [[Yellow]]col:42[[RESET]] invalid [[Green]]'int'[[RESET]]
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[BLUE]]NoInlineAttr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:18[[RESET]]>
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]line:28:8[[RESET]]> [[Yellow]]col:8[[RESET]] implicit used constexpr[[CYAN]] Invalid[[RESET]] [[Green]]'void () noexcept'[[RESET]] inline
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[MAGENTA]]CompoundStmt[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]>
//CHECK: {{^}}[[Blue]]| |-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Yellow]]col:8[[RESET]] implicit constexpr[[CYAN]] Invalid[[RESET]] [[Green]]'void (const Invalid &)'[[RESET]] inline default trivial noexcept-unevaluated 0x{{[0-9a-fA-F]*}}
//CHECK: {{^}}[[Blue]]| | `-[[RESET]][[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Yellow]]col:8[[RESET]] [[Green]]'const Invalid &'[[RESET]]
//CHECK: {{^}}[[Blue]]| `-[[RESET]][[GREEN]]CXXConstructorDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Yellow]]col:8[[RESET]] implicit constexpr[[CYAN]] Invalid[[RESET]] [[Green]]'void (Invalid &&)'[[RESET]] inline default trivial noexcept-unevaluated 0x{{[0-9a-fA-F]*}}
//CHECK: {{^}}[[Blue]]|   `-[[RESET]][[GREEN]]ParmVarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:8[[RESET]]> [[Yellow]]col:8[[RESET]] [[Green]]'Invalid &&'[[RESET]]
//CHECK: {{^}}[[Blue]]`-[[RESET]][[GREEN]]VarDecl[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:1[[RESET]], [[Yellow]]line:30:3[[RESET]]> [[Yellow]]col:3[[RESET]][[CYAN]] Invalid[[RESET]] [[Green]]'struct Invalid':'Invalid'[[RESET]]
//CHECK: {{^}}[[Blue]]  `-[[RESET]][[MAGENTA]]CXXConstructExpr[[RESET]][[Yellow]] 0x{{[0-9a-fA-F]*}}[[RESET]] <[[Yellow]]col:3[[RESET]]> [[Green]]'struct Invalid':'Invalid'[[RESET]][[Cyan]][[RESET]][[Cyan]][[RESET]] [[Green]]'void () noexcept'[[RESET]]
