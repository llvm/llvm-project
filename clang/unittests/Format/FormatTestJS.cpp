//===- unittest/Format/FormatTestJS.cpp - Formatting unit tests for JS ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FormatTestUtils.h"
#include "clang/Format/Format.h"
#include "llvm/Support/Debug.h"
#include "gtest/gtest.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {

class FormatTestJS : public ::testing::Test {
protected:
  static std::string format(llvm::StringRef Code, unsigned Offset,
                            unsigned Length, const FormatStyle &Style) {
    DEBUG(llvm::errs() << "---\n");
    DEBUG(llvm::errs() << Code << "\n\n");
    std::vector<tooling::Range> Ranges(1, tooling::Range(Offset, Length));
    bool IncompleteFormat = false;
    tooling::Replacements Replaces =
        reformat(Style, Code, Ranges, "<stdin>", &IncompleteFormat);
    EXPECT_FALSE(IncompleteFormat);
    auto Result = applyAllReplacements(Code, Replaces);
    EXPECT_TRUE(static_cast<bool>(Result));
    DEBUG(llvm::errs() << "\n" << *Result << "\n\n");
    return *Result;
  }

  static std::string format(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    return format(Code, 0, Code.size(), Style);
  }

  static FormatStyle getGoogleJSStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }

  static void verifyFormat(
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    std::string Result = format(test::messUp(Code), Style);
    EXPECT_EQ(Code.str(), Result) << "Formatted:\n" << Result;
  }

  static void verifyFormat(
      llvm::StringRef Expected,
      llvm::StringRef Code,
      const FormatStyle &Style = getGoogleStyle(FormatStyle::LK_JavaScript)) {
    std::string Result = format(Code, Style);
    EXPECT_EQ(Expected.str(), Result) << "Formatted:\n" << Result;
  }
};

TEST_F(FormatTestJS, BlockComments) {
  verifyFormat("/* aaaaaaaaaaaaa */ aaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa);");
}

TEST_F(FormatTestJS, UnderstandsJavaScriptOperators) {
  verifyFormat("a == = b;");
  verifyFormat("a != = b;");

  verifyFormat("a === b;");
  verifyFormat("aaaaaaa ===\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a !== b;");
  verifyFormat("aaaaaaa !==\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d !==\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));

  verifyFormat("a >> >= b;");

  verifyFormat("a >>> b;");
  verifyFormat("aaaaaaa >>>\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("a >>>= b;");
  verifyFormat("aaaaaaa >>>=\n    b;", getGoogleJSStyleWithColumns(10));
  verifyFormat("if (a + b + c +\n"
               "        d >>>\n"
               "    e + f + g)\n"
               "  q();",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("var x = aaaaaaaaaa ?\n"
               "    bbbbbb :\n"
               "    ccc;",
               getGoogleJSStyleWithColumns(20));

  verifyFormat("var b = a.map((x) => x + 1);");
  verifyFormat("return ('aaa') in bbbb;");
  verifyFormat("var x = aaaaaaaaaaaaaaaaaaaaaaaaa() in\n"
               "    aaaa.aaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;");
  FormatStyle Style = getGoogleJSStyleWithColumns(80);
  Style.AlignOperands = true;
  verifyFormat("var x = aaaaaaaaaaaaaaaaaaaaaaaaa() in\n"
               "        aaaa.aaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_All;
  verifyFormat("var x = aaaaaaaaaaaaaaaaaaaaaaaaa()\n"
               "            in aaaa.aaaaaa.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               Style);

  // ES6 spread operator.
  verifyFormat("someFunction(...a);");
  verifyFormat("var x = [1, ...a, 2];");
}

TEST_F(FormatTestJS, UnderstandsAmpAmp) {
  verifyFormat("e && e.SomeFunction();");
}

TEST_F(FormatTestJS, LiteralOperatorsCanBeKeywords) {
  verifyFormat("not.and.or.not_eq = 1;");
}

TEST_F(FormatTestJS, ReservedWords) {
  // JavaScript reserved words (aka keywords) are only illegal when used as
  // Identifiers, but are legal as IdentifierNames.
  verifyFormat("x.class.struct = 1;");
  verifyFormat("x.case = 1;");
  verifyFormat("x.interface = 1;");
  verifyFormat("x.for = 1;");
  verifyFormat("x.of() = 1;");
  verifyFormat("x.in() = 1;");
  verifyFormat("x.let() = 1;");
  verifyFormat("x.var() = 1;");
  verifyFormat("x = {\n"
               "  a: 12,\n"
               "  interface: 1,\n"
               "  switch: 1,\n"
               "};");
  verifyFormat("var struct = 2;");
  verifyFormat("var union = 2;");
  verifyFormat("var interface = 2;");
  verifyFormat("interface = 2;");
  verifyFormat("x = interface instanceof y;");
}

TEST_F(FormatTestJS, CppKeywords) {
  // Make sure we don't mess stuff up because of C++ keywords.
  verifyFormat("return operator && (aa);");
}

TEST_F(FormatTestJS, ES6DestructuringAssignment) {
  verifyFormat("var [a, b, c] = [1, 2, 3];");
  verifyFormat("const [a, b, c] = [1, 2, 3];");
  verifyFormat("let [a, b, c] = [1, 2, 3];");
  verifyFormat("var {a, b} = {a: 1, b: 2};");
  verifyFormat("let {a, b} = {a: 1, b: 2};");
}

TEST_F(FormatTestJS, ContainerLiterals) {
  verifyFormat("var x = {\n"
               "  y: function(a) {\n"
               "    return a;\n"
               "  }\n"
               "};");
  verifyFormat("return {\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("return {\n"
               "  a: a,\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("return {\n"
               "  a: a,\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  },\n"
               "  link: function() {\n"
               "    f();  //\n"
               "  }\n"
               "};");
  verifyFormat("var stuff = {\n"
               "  // comment for update\n"
               "  update: false,\n"
               "  // comment for modules\n"
               "  modules: false,\n"
               "  // comment for tasks\n"
               "  tasks: false\n"
               "};");
  verifyFormat("return {\n"
               "  'finish':\n"
               "      //\n"
               "      a\n"
               "};");
  verifyFormat("var obj = {\n"
               "  fooooooooo: function(x) {\n"
               "    return x.zIsTooLongForOneLineWithTheDeclarationLine();\n"
               "  }\n"
               "};");
  // Simple object literal, as opposed to enum style below.
  verifyFormat("var obj = {a: 123};");
  // Enum style top level assignment.
  verifyFormat("X = {\n  a: 123\n};");
  verifyFormat("X.Y = {\n  a: 123\n};");
  // But only on the top level, otherwise its a plain object literal assignment.
  verifyFormat("function x() {\n"
               "  y = {z: 1};\n"
               "}");
  verifyFormat("x = foo && {a: 123};");

  // Arrow functions in object literals.
  verifyFormat("var x = {\n"
               "  y: (a) => {\n"
               "    return a;\n"
               "  }\n"
               "};");
  verifyFormat("var x = {y: (a) => a};");

  // Computed keys.
  verifyFormat("var x = {[a]: 1, b: 2, [c]: 3};");
  verifyFormat("var x = {\n"
               "  [a]: 1,\n"
               "  b: 2,\n"
               "  [c]: 3,\n"
               "};");

  // Object literals can leave out labels.
  verifyFormat("f({a}, () => {\n"
               "  g();  //\n"
               "});");

  // Keys can be quoted.
  verifyFormat("var x = {\n"
               "  a: a,\n"
               "  b: b,\n"
               "  'c': c,\n"
               "};");
}

TEST_F(FormatTestJS, MethodsInObjectLiterals) {
  verifyFormat("var o = {\n"
               "  value: 'test',\n"
               "  get value() {  // getter\n"
               "    return this.value;\n"
               "  }\n"
               "};");
  verifyFormat("var o = {\n"
               "  value: 'test',\n"
               "  set value(val) {  // setter\n"
               "    this.value = val;\n"
               "  }\n"
               "};");
  verifyFormat("var o = {\n"
               "  value: 'test',\n"
               "  someMethod(val) {  // method\n"
               "    doSomething(this.value + val);\n"
               "  }\n"
               "};");
  verifyFormat("var o = {\n"
               "  someMethod(val) {  // method\n"
               "    doSomething(this.value + val);\n"
               "  },\n"
               "  someOtherMethod(val) {  // method\n"
               "    doSomething(this.value + val);\n"
               "  }\n"
               "};");
}

TEST_F(FormatTestJS, SpacesInContainerLiterals) {
  verifyFormat("var arr = [1, 2, 3];");
  verifyFormat("f({a: 1, b: 2, c: 3});");

  verifyFormat("var object_literal_with_long_name = {\n"
               "  a: 'aaaaaaaaaaaaaaaaaa',\n"
               "  b: 'bbbbbbbbbbbbbbbbbb'\n"
               "};");

  verifyFormat("f({a: 1, b: 2, c: 3});",
               getChromiumStyle(FormatStyle::LK_JavaScript));
  verifyFormat("f({'a': [{}]});");
}

TEST_F(FormatTestJS, SingleQuotedStrings) {
  verifyFormat("this.function('', true);");
}

TEST_F(FormatTestJS, GoogScopes) {
  verifyFormat("goog.scope(function() {\n"
               "var x = a.b;\n"
               "var y = c.d;\n"
               "});  // goog.scope");
  verifyFormat("goog.scope(function() {\n"
               "// test\n"
               "var x = 0;\n"
               "// test\n"
               "});");
}

TEST_F(FormatTestJS, GoogModules) {
  verifyFormat("goog.module('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.require('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.provide('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("var long = goog.require('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.setTestOnly('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("goog.forwardDeclare('this.is.really.absurdly.long');",
               getGoogleJSStyleWithColumns(40));

  // These should be wrapped normally.
  verifyFormat(
      "var MyLongClassName =\n"
      "    goog.module.get('my.long.module.name.followedBy.MyLongClassName');");
}

TEST_F(FormatTestJS, FormatsNamespaces) {
  verifyFormat("namespace Foo {\n"
               "  export let x = 1;\n"
               "}\n");
  verifyFormat("declare namespace Foo {\n"
               "  export let x: number;\n"
               "}\n");
}

TEST_F(FormatTestJS, FormatsFreestandingFunctions) {
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) {\n"
               "    return a;\n"
               "  }\n"
               "  inner1(a, b);\n"
               "}\n"
               "function outer2(a, b) {\n"
               "  function inner2(a, b) {\n"
               "    return a;\n"
               "  }\n"
               "  inner2(a, b);\n"
               "}");
  verifyFormat("function f() {}");
}

TEST_F(FormatTestJS, GeneratorFunctions) {
  verifyFormat("function* f() {\n"
               "  let x = 1;\n"
               "  yield x;\n"
               "  yield* something();\n"
               "}");
  verifyFormat("function*\n"
               "    f() {\n"
               "}",
               getGoogleJSStyleWithColumns(8));
  verifyFormat("export function* f() {\n"
               "  yield 1;\n"
               "}\n");
  verifyFormat("class X {\n"
               "  * generatorMethod() {\n"
               "    yield x;\n"
               "  }\n"
               "}");
}

TEST_F(FormatTestJS, AsyncFunctions) {
  verifyFormat("async function f() {\n"
               "  let x = 1;\n"
               "  return fetch(x);\n"
               "}");
  verifyFormat("async function* f() {\n"
               "  yield fetch(x);\n"
               "}");
  verifyFormat("export async function f() {\n"
               "  return fetch(x);\n"
               "}");
  verifyFormat("class X {\n"
               "  async asyncMethod() {\n"
               "    return fetch(1);\n"
               "  }\n"
               "}");
  verifyFormat("function initialize() {\n"
               "  // Comment.\n"
               "  return async.then();\n"
               "}\n");
}

TEST_F(FormatTestJS, ArrayLiterals) {
  verifyFormat("var aaaaa: List<SomeThing> =\n"
               "    [new SomeThingAAAAAAAAAAAA(), new SomeThingBBBBBBBBB()];");
  verifyFormat("return [\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "  ccccccccccccccccccccccccccc\n"
               "];");
  verifyFormat("return [\n"
               "  aaaa().bbbbbbbb('A'),\n"
               "  aaaa().bbbbbbbb('B'),\n"
               "  aaaa().bbbbbbbb('C'),\n"
               "];");
  verifyFormat("var someVariable = SomeFunction([\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "  ccccccccccccccccccccccccccc\n"
               "]);");
  verifyFormat("var someVariable = SomeFunction([\n"
               "  [aaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbb],\n"
               "]);",
               getGoogleJSStyleWithColumns(51));
  verifyFormat("var someVariable = SomeFunction(aaaa, [\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "  ccccccccccccccccccccccccccc\n"
               "]);");
  verifyFormat("var someVariable = SomeFunction(\n"
               "    aaaa,\n"
               "    [\n"
               "      aaaaaaaaaaaaaaaaaaaaaaaaaa, bbbbbbbbbbbbbbbbbbbbbbbbbb,\n"
               "      cccccccccccccccccccccccccc\n"
               "    ],\n"
               "    aaaa);");
  verifyFormat("var aaaa = aaaaa ||  // wrap\n"
               "    [];");

  verifyFormat("someFunction([], {a: a});");
}

TEST_F(FormatTestJS, ColumnLayoutForArrayLiterals) {
  verifyFormat("var array = [\n"
               "  a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,\n"
               "  a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,\n"
               "];");
  verifyFormat("var array = someFunction([\n"
               "  a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,\n"
               "  a, a, a, a, a, a, a, a, a, a, a, a, a, a, a,\n"
               "]);");
}

TEST_F(FormatTestJS, FunctionLiterals) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("doFoo(function() {});");
  verifyFormat("doFoo(function() { return 1; });", Style);
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};");
  verifyFormat("var func =  //\n"
               "    function() {\n"
               "  return 1;\n"
               "};");
  verifyFormat("return {\n"
               "  body: {\n"
               "    setAttribute: function(key, val) { this[key] = val; },\n"
               "    getAttribute: function(key) { return this[key]; },\n"
               "    style: {direction: ''}\n"
               "  }\n"
               "};",
               Style);
  verifyFormat("abc = xyz ? function() {\n"
               "  return 1;\n"
               "} : function() {\n"
               "  return -1;\n"
               "};");

  verifyFormat("var closure = goog.bind(\n"
               "    function() {  // comment\n"
               "      foo();\n"
               "      bar();\n"
               "    },\n"
               "    this, arg1IsReallyLongAndNeeedsLineBreaks,\n"
               "    arg3IsReallyLongAndNeeedsLineBreaks);");
  verifyFormat("var closure = goog.bind(function() {  // comment\n"
               "  foo();\n"
               "  bar();\n"
               "}, this);");
  verifyFormat("return {\n"
               "  a: 'E',\n"
               "  b: function() {\n"
               "    return function() {\n"
               "      f();  //\n"
               "    };\n"
               "  }\n"
               "};");
  verifyFormat("{\n"
               "  var someVariable = function(x) {\n"
               "    return x.zIsTooLongForOneLineWithTheDeclarationLine();\n"
               "  };\n"
               "}");
  verifyFormat("someLooooooooongFunction(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    function(aaaaaaaaaaaaaaaaaaaaaaaaaaaaa) {\n"
               "      // code\n"
               "    });");

  verifyFormat("return {\n"
               "  a: function SomeFunction() {\n"
               "    // ...\n"
               "    return 1;\n"
               "  }\n"
               "};");
  verifyFormat("this.someObject.doSomething(aaaaaaaaaaaaaaaaaaaaaaaaaa)\n"
               "    .then(goog.bind(function(aaaaaaaaaaa) {\n"
               "      someFunction();\n"
               "      someFunction();\n"
               "    }, this), aaaaaaaaaaaaaaaaa);");

  verifyFormat("someFunction(goog.bind(function() {\n"
               "  doSomething();\n"
               "  doSomething();\n"
               "}, this), goog.bind(function() {\n"
               "  doSomething();\n"
               "  doSomething();\n"
               "}, this));");

  // FIXME: This is bad, we should be wrapping before "function() {".
  verifyFormat("someFunction(function() {\n"
               "  doSomething();  // break\n"
               "})\n"
               "    .doSomethingElse(\n"
               "        // break\n"
               "        );");

  Style.ColumnLimit = 33;
  verifyFormat("f({a: function() { return 1; }});", Style);
  Style.ColumnLimit = 32;
  verifyFormat("f({\n"
               "  a: function() { return 1; }\n"
               "});",
               Style);

}

TEST_F(FormatTestJS, InliningFunctionLiterals) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};",
               Style);
  verifyFormat("var func = doSomething(function() { return 1; });", Style);
  verifyFormat("var outer = function() {\n"
               "  var inner = function() { return 1; }\n"
               "};",
               Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) { return a; }\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("var func = function() { return 1; };", Style);
  verifyFormat("var func = doSomething(function() { return 1; });", Style);
  verifyFormat(
      "var outer = function() { var inner = function() { return 1; } };",
      Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) { return a; }\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_None;
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};",
               Style);
  verifyFormat("var func = doSomething(function() {\n"
               "  return 1;\n"
               "});",
               Style);
  verifyFormat("var outer = function() {\n"
               "  var inner = function() {\n"
               "    return 1;\n"
               "  }\n"
               "};",
               Style);
  verifyFormat("function outer1(a, b) {\n"
               "  function inner1(a, b) {\n"
               "    return a;\n"
               "  }\n"
               "}",
               Style);

  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Empty;
  verifyFormat("var func = function() {\n"
               "  return 1;\n"
               "};",
               Style);
}

TEST_F(FormatTestJS, MultipleFunctionLiterals) {
  FormatStyle Style = getGoogleStyle(FormatStyle::LK_JavaScript);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_All;
  verifyFormat("promise.then(\n"
               "    function success() {\n"
               "      doFoo();\n"
               "      doBar();\n"
               "    },\n"
               "    function error() {\n"
               "      doFoo();\n"
               "      doBaz();\n"
               "    },\n"
               "    []);\n");
  verifyFormat("promise.then(\n"
               "    function success() {\n"
               "      doFoo();\n"
               "      doBar();\n"
               "    },\n"
               "    [],\n"
               "    function error() {\n"
               "      doFoo();\n"
               "      doBaz();\n"
               "    });\n");
  verifyFormat("promise.then(\n"
               "    [],\n"
               "    function success() {\n"
               "      doFoo();\n"
               "      doBar();\n"
               "    },\n"
               "    function error() {\n"
               "      doFoo();\n"
               "      doBaz();\n"
               "    });\n");

  verifyFormat("getSomeLongPromise()\n"
               "    .then(function(value) { body(); })\n"
               "    .thenCatch(function(error) {\n"
               "      body();\n"
               "      body();\n"
               "    });",
               Style);
  verifyFormat("getSomeLongPromise()\n"
               "    .then(function(value) {\n"
               "      body();\n"
               "      body();\n"
               "    })\n"
               "    .thenCatch(function(error) {\n"
               "      body();\n"
               "      body();\n"
               "    });");

  verifyFormat("getSomeLongPromise()\n"
               "    .then(function(value) { body(); })\n"
               "    .thenCatch(function(error) { body(); });",
               Style);

  verifyFormat("return [aaaaaaaaaaaaaaaaaaaaaa]\n"
               "    .aaaaaaa(function() {\n"
               "      //\n"
               "    })\n"
               "    .bbbbbb();");
}

TEST_F(FormatTestJS, ArrowFunctions) {
  verifyFormat("var x = (a) => {\n"
               "  return a;\n"
               "};");
  verifyFormat("var x = (a) => {\n"
               "  function y() {\n"
               "    return 42;\n"
               "  }\n"
               "  return a;\n"
               "};");
  verifyFormat("var x = (a: type): {some: type} => {\n"
               "  return a;\n"
               "};");
  verifyFormat("var x = (a) => a;");
  verifyFormat("return () => [];");
  verifyFormat("var aaaaaaaaaaaaaaaaaaaa = {\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaa:\n"
               "      (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "       aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa) =>\n"
               "          aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "};");
  verifyFormat("var a = a.aaaaaaa(\n"
               "    (a: a) => aaaaaaaaaaaaaaaaaaaaaaaaa(bbbbbbbbb) &&\n"
               "        aaaaaaaaaaaaaaaaaaaaaaaaa(bbbbbbb));");
  verifyFormat("var a = a.aaaaaaa(\n"
               "    (a: a) => aaaaaaaaaaaaaaaaaaaaa(bbbbbbbbb) ?\n"
               "        aaaaaaaaaaaaaaaaaaaaa(bbbbbbb) :\n"
               "        aaaaaaaaaaaaaaaaaaaaa(bbbbbbb));");

  // FIXME: This is bad, we should be wrapping before "() => {".
  verifyFormat("someFunction(() => {\n"
               "  doSomething();  // break\n"
               "})\n"
               "    .doSomethingElse(\n"
               "        // break\n"
               "        );");
}

TEST_F(FormatTestJS, ReturnStatements) {
  verifyFormat("function() {\n"
               "  return [hello, world];\n"
               "}");
}

TEST_F(FormatTestJS, ForLoops) {
  verifyFormat("for (var i in [2, 3]) {\n"
               "}");
  verifyFormat("for (var i of [2, 3]) {\n"
               "}");
  verifyFormat("for (let {a, b} of x) {\n"
               "}");
  verifyFormat("for (let {a, b} in x) {\n"
               "}");
}

TEST_F(FormatTestJS, WrapRespectsAutomaticSemicolonInsertion) {
  // The following statements must not wrap, as otherwise the program meaning
  // would change due to automatic semicolon insertion.
  // See http://www.ecma-international.org/ecma-262/5.1/#sec-7.9.1.
  verifyFormat("return aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("return /* hello! */ aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("continue aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("continue /* hello! */ aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("break aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("throw aaaaa;", getGoogleJSStyleWithColumns(10));
  verifyFormat("aaaaaaaaa++;", getGoogleJSStyleWithColumns(10));
  verifyFormat("aaaaaaaaa--;", getGoogleJSStyleWithColumns(10));
  verifyFormat("return [\n"
               "  aaa\n"
               "];",
               getGoogleJSStyleWithColumns(12));
}

TEST_F(FormatTestJS, AutomaticSemicolonInsertionHeuristic) {
  verifyFormat("a\n"
               "b;",
               " a \n"
               " b ;");
  verifyFormat("a()\n"
               "b;",
               " a ()\n"
               " b ;");
  verifyFormat("a[b]\n"
               "c;",
               "a [b]\n"
               "c ;");
  verifyFormat("1\n"
               "a;",
               "1 \n"
               "a ;");
  verifyFormat("a\n"
               "1;",
               "a \n"
               "1 ;");
  verifyFormat("a\n"
               "'x';",
               "a \n"
               " 'x';");
  verifyFormat("a++\n"
               "b;",
               "a ++\n"
               "b ;");
  verifyFormat("a\n"
               "!b && c;",
               "a \n"
               " ! b && c;");
  verifyFormat("a\n"
               "if (1) f();",
               " a\n"
               " if (1) f();");
  verifyFormat("a\n"
               "class X {}",
               " a\n"
               " class X {}");
  verifyFormat("var a", "var\n"
                        "a");
  verifyFormat("x instanceof String", "x\n"
                                      "instanceof\n"
                                      "String");
  verifyFormat("function f(@Foo bar) {}", "function f(@Foo\n"
                                          "  bar) {}");
}

TEST_F(FormatTestJS, ClosureStyleCasts) {
  verifyFormat("var x = /** @type {foo} */ (bar);");
}

TEST_F(FormatTestJS, TryCatch) {
  verifyFormat("try {\n"
               "  f();\n"
               "} catch (e) {\n"
               "  g();\n"
               "} finally {\n"
               "  h();\n"
               "}");

  // But, of course, "catch" is a perfectly fine function name in JavaScript.
  verifyFormat("someObject.catch();");
  verifyFormat("someObject.new();");
  verifyFormat("someObject.delete();");
}

TEST_F(FormatTestJS, StringLiteralConcatenation) {
  verifyFormat("var literal = 'hello ' +\n"
               "    'world';");
}

TEST_F(FormatTestJS, RegexLiteralClassification) {
  // Regex literals.
  verifyFormat("var regex = /abc/;");
  verifyFormat("f(/abc/);");
  verifyFormat("f(abc, /abc/);");
  verifyFormat("some_map[/abc/];");
  verifyFormat("var x = a ? /abc/ : /abc/;");
  verifyFormat("for (var i = 0; /abc/.test(s[i]); i++) {\n}");
  verifyFormat("var x = !/abc/.test(y);");
  verifyFormat("var x = a && /abc/.test(y);");
  verifyFormat("var x = a || /abc/.test(y);");
  verifyFormat("var x = a + /abc/.search(y);");
  verifyFormat("/abc/.search(y);");
  verifyFormat("var regexs = {/abc/, /abc/};");
  verifyFormat("return /abc/;");

  // Not regex literals.
  verifyFormat("var a = a / 2 + b / 3;");
  verifyFormat("var a = a++ / 2;");
  // Prefix unary can operate on regex literals, not that it makes sense.
  verifyFormat("var a = ++/a/;");

  // This is a known issue, regular expressions are incorrectly detected if
  // directly following a closing parenthesis.
  verifyFormat("if (foo) / bar /.exec(baz);");
}

TEST_F(FormatTestJS, RegexLiteralSpecialCharacters) {
  verifyFormat("var regex = /=/;");
  verifyFormat("var regex = /a*/;");
  verifyFormat("var regex = /a+/;");
  verifyFormat("var regex = /a?/;");
  verifyFormat("var regex = /.a./;");
  verifyFormat("var regex = /a\\*/;");
  verifyFormat("var regex = /^a$/;");
  verifyFormat("var regex = /\\/a/;");
  verifyFormat("var regex = /(?:x)/;");
  verifyFormat("var regex = /x(?=y)/;");
  verifyFormat("var regex = /x(?!y)/;");
  verifyFormat("var regex = /x|y/;");
  verifyFormat("var regex = /a{2}/;");
  verifyFormat("var regex = /a{1,3}/;");

  verifyFormat("var regex = /[abc]/;");
  verifyFormat("var regex = /[^abc]/;");
  verifyFormat("var regex = /[\\b]/;");
  verifyFormat("var regex = /[/]/;");
  verifyFormat("var regex = /[\\/]/;");
  verifyFormat("var regex = /\\[/;");
  verifyFormat("var regex = /\\\\[/]/;");
  verifyFormat("var regex = /}[\"]/;");
  verifyFormat("var regex = /}[/\"]/;");
  verifyFormat("var regex = /}[\"/]/;");

  verifyFormat("var regex = /\\b/;");
  verifyFormat("var regex = /\\B/;");
  verifyFormat("var regex = /\\d/;");
  verifyFormat("var regex = /\\D/;");
  verifyFormat("var regex = /\\f/;");
  verifyFormat("var regex = /\\n/;");
  verifyFormat("var regex = /\\r/;");
  verifyFormat("var regex = /\\s/;");
  verifyFormat("var regex = /\\S/;");
  verifyFormat("var regex = /\\t/;");
  verifyFormat("var regex = /\\v/;");
  verifyFormat("var regex = /\\w/;");
  verifyFormat("var regex = /\\W/;");
  verifyFormat("var regex = /a(a)\\1/;");
  verifyFormat("var regex = /\\0/;");
  verifyFormat("var regex = /\\\\/g;");
  verifyFormat("var regex = /\\a\\\\/g;");
  verifyFormat("var regex = /\a\\//g;");
  verifyFormat("var regex = /a\\//;\n"
               "var x = 0;");
  verifyFormat("var regex = /'/g;", "var regex = /'/g ;");
  verifyFormat("var regex = /'/g;  //'", "var regex = /'/g ; //'");
  verifyFormat("var regex = /\\/*/;\n"
               "var x = 0;",
               "var regex = /\\/*/;\n"
               "var x=0;");
  verifyFormat("var x = /a\\//;", "var x = /a\\//  \n;");
  verifyFormat("var regex = /\"/;", getGoogleJSStyleWithColumns(16));
  verifyFormat("var regex =\n"
               "    /\"/;",
               getGoogleJSStyleWithColumns(15));
  verifyFormat("var regex =  //\n"
               "    /a/;");
  verifyFormat("var regexs = [\n"
               "  /d/,   //\n"
               "  /aa/,  //\n"
               "];");
}

TEST_F(FormatTestJS, RegexLiteralModifiers) {
  verifyFormat("var regex = /abc/g;");
  verifyFormat("var regex = /abc/i;");
  verifyFormat("var regex = /abc/m;");
  verifyFormat("var regex = /abc/y;");
}

TEST_F(FormatTestJS, RegexLiteralLength) {
  verifyFormat("var regex = /aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(60));
  verifyFormat("var regex =\n"
               "    /aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(60));
  verifyFormat("var regex = /\\xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/;",
               getGoogleJSStyleWithColumns(50));
}

TEST_F(FormatTestJS, RegexLiteralExamples) {
  verifyFormat("var regex = search.match(/(?:\?|&)times=([^?&]+)/i);");
}

TEST_F(FormatTestJS, TypeAnnotations) {
  verifyFormat("var x: string;");
  verifyFormat("var x: {a: string; b: number;} = {};");
  verifyFormat("function x(): string {\n  return 'x';\n}");
  verifyFormat("function x(): {x: string} {\n  return {x: 'x'};\n}");
  verifyFormat("function x(y: string): string {\n  return 'x';\n}");
  verifyFormat("for (var y: string in x) {\n  x();\n}");
  verifyFormat("for (var y: string of x) {\n  x();\n}");
  verifyFormat("function x(y: {a?: number;} = {}): number {\n"
               "  return 12;\n"
               "}");
  verifyFormat("((a: string, b: number): string => a + b);");
  verifyFormat("var x: (y: number) => string;");
  verifyFormat("var x: P<string, (a: number) => string>;");
  verifyFormat("var x = {\n"
               "  y: function(): z {\n"
               "    return 1;\n"
               "  }\n"
               "};");
  verifyFormat("var x = {\n"
               "  y: function(): {a: number} {\n"
               "    return 1;\n"
               "  }\n"
               "};");
  verifyFormat("function someFunc(args: string[]):\n"
               "    {longReturnValue: string[]} {}",
               getGoogleJSStyleWithColumns(60));
}

TEST_F(FormatTestJS, UnionIntersectionTypes) {
  verifyFormat("let x: A|B = A | B;");
  verifyFormat("let x: A&B|C = A & B;");
  verifyFormat("let x: Foo<A|B> = new Foo<A|B>();");
  verifyFormat("function(x: A|B): C&D {}");
  verifyFormat("function(x: A|B = A | B): C&D {}");
  verifyFormat("function x(path: number|string) {}");
  verifyFormat("function x(): string|number {}");
  verifyFormat("type Foo = Bar|Baz;");
  verifyFormat("type Foo = Bar<X>|Baz;");
  verifyFormat("type Foo = (Bar<X>|Baz);");
  verifyFormat("let x: Bar|Baz;");
  verifyFormat("let x: Bar<X>|Baz;");
  verifyFormat("let x: (Foo|Bar)[];");
}

TEST_F(FormatTestJS, ClassDeclarations) {
  verifyFormat("class C {\n  x: string = 12;\n}");
  verifyFormat("class C {\n  x(): string => 12;\n}");
  verifyFormat("class C {\n  ['x' + 2]: string = 12;\n}");
  verifyFormat("class C {\n  private x: string = 12;\n}");
  verifyFormat("class C {\n  private static x: string = 12;\n}");
  verifyFormat("class C {\n  static x(): string {\n    return 'asd';\n  }\n}");
  verifyFormat("class C extends P implements I {}");
  verifyFormat("class C extends p.P implements i.I {}");
  verifyFormat("class Test {\n"
               "  aaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaa: aaaaaaaaaaaaaaaaaaaa):\n"
               "      aaaaaaaaaaaaaaaaaaaaaa {}\n"
               "}");
  verifyFormat("foo = class Name {\n"
               "  constructor() {}\n"
               "};");
  verifyFormat("foo = class {\n"
               "  constructor() {}\n"
               "};");
  verifyFormat("class C {\n"
               "  x: {y: Z;} = {};\n"
               "  private y: {y: Z;} = {};\n"
               "}");

  // ':' is not a type declaration here.
  verifyFormat("class X {\n"
               "  subs = {\n"
               "    'b': {\n"
               "      'c': 1,\n"
               "    },\n"
               "  };\n"
               "}");
  verifyFormat("@Component({\n"
               "  moduleId: module.id,\n"
               "})\n"
               "class SessionListComponent implements OnDestroy, OnInit {\n"
               "}");
}

TEST_F(FormatTestJS, InterfaceDeclarations) {
  verifyFormat("interface I {\n"
               "  x: string;\n"
               "  enum: string[];\n"
               "  enum?: string[];\n"
               "}\n"
               "var y;");
  // Ensure that state is reset after parsing the interface.
  verifyFormat("interface a {}\n"
               "export function b() {}\n"
               "var x;");

  // Arrays of object type literals.
  verifyFormat("interface I {\n"
               "  o: {}[];\n"
               "}");
}

TEST_F(FormatTestJS, EnumDeclarations) {
  verifyFormat("enum Foo {\n"
               "  A = 1,\n"
               "  B\n"
               "}");
  verifyFormat("export /* somecomment*/ enum Foo {\n"
               "  A = 1,\n"
               "  B\n"
               "}");
  verifyFormat("enum Foo {\n"
               "  A = 1,  // comment\n"
               "  B\n"
               "}\n"
               "var x = 1;");
}

TEST_F(FormatTestJS, MetadataAnnotations) {
  verifyFormat("@A\nclass C {\n}");
  verifyFormat("@A({arg: 'value'})\nclass C {\n}");
  verifyFormat("@A\n@B\nclass C {\n}");
  verifyFormat("class C {\n  @A x: string;\n}");
  verifyFormat("class C {\n"
               "  @A\n"
               "  private x(): string {\n"
               "    return 'y';\n"
               "  }\n"
               "}");
  verifyFormat("class C {\n"
               "  private x(@A x: string) {}\n"
               "}");
  verifyFormat("class X {}\n"
               "class Y {}");
}

TEST_F(FormatTestJS, TypeAliases) {
  verifyFormat("type X = number;\n"
               "class C {}");
  verifyFormat("type X<Y> = Z<Y>;");
  verifyFormat("type X = {\n"
               "  y: number\n"
               "};\n"
               "class C {}");
}

TEST_F(FormatTestJS, Modules) {
  verifyFormat("import SomeThing from 'some/module.js';");
  verifyFormat("import {X, Y} from 'some/module.js';");
  verifyFormat("import a, {X, Y} from 'some/module.js';");
  verifyFormat("import {X, Y,} from 'some/module.js';");
  verifyFormat("import {X as myLocalX, Y as myLocalY} from 'some/module.js';");
  // Ensure Automatic Semicolon Insertion does not break on "as\n".
  verifyFormat("import {X as myX} from 'm';", "import {X as\n"
                                              " myX} from 'm';");
  verifyFormat("import * as lib from 'some/module.js';");
  verifyFormat("var x = {import: 1};\nx.import = 2;");

  verifyFormat("export function fn() {\n"
               "  return 'fn';\n"
               "}");
  verifyFormat("export function A() {}\n"
               "export default function B() {}\n"
               "export function C() {}");
  verifyFormat("export default () => {\n"
               "  let x = 1;\n"
               "  return x;\n"
               "}");
  verifyFormat("export const x = 12;");
  verifyFormat("export default class X {}");
  verifyFormat("export {X, Y} from 'some/module.js';");
  verifyFormat("export {X, Y,} from 'some/module.js';");
  verifyFormat("export {SomeVeryLongExport as X, "
               "SomeOtherVeryLongExport as Y} from 'some/module.js';");
  // export without 'from' is wrapped.
  verifyFormat("export let someRatherLongVariableName =\n"
               "    someSurprisinglyLongVariable + someOtherRatherLongVar;");
  // ... but not if from is just an identifier.
  verifyFormat("export {\n"
               "  from as from,\n"
               "  someSurprisinglyLongVariable as\n"
               "      from\n"
               "};",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("export class C {\n"
               "  x: number;\n"
               "  y: string;\n"
               "}");
  verifyFormat("export class X { y: number; }");
  verifyFormat("export abstract class X { y: number; }");
  verifyFormat("export default class X { y: number }");
  verifyFormat("export default function() {\n  return 1;\n}");
  verifyFormat("export var x = 12;");
  verifyFormat("class C {}\n"
               "export function f() {}\n"
               "var v;");
  verifyFormat("export var x: number = 12;");
  verifyFormat("export const y = {\n"
               "  a: 1,\n"
               "  b: 2\n"
               "};");
  verifyFormat("export enum Foo {\n"
               "  BAR,\n"
               "  // adsdasd\n"
               "  BAZ\n"
               "}");
  verifyFormat("export default [\n"
               "  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
               "];");
  verifyFormat("export default [];");
  verifyFormat("export default () => {};");
  verifyFormat("export interface Foo { foo: number; }\n"
               "export class Bar {\n"
               "  blah(): string {\n"
               "    return this.blah;\n"
               "  };\n"
               "}");
}

TEST_F(FormatTestJS, ImportWrapping) {
  verifyFormat("import {VeryLongImportsAreAnnoying, VeryLongImportsAreAnnoying,"
               " VeryLongImportsAreAnnoying, VeryLongImportsAreAnnoying"
               "} from 'some/module.js';");
  FormatStyle Style = getGoogleJSStyleWithColumns(80);
  Style.JavaScriptWrapImports = true;
  verifyFormat("import {\n"
               "  VeryLongImportsAreAnnoying,\n"
               "  VeryLongImportsAreAnnoying,\n"
               "  VeryLongImportsAreAnnoying,\n"
               "} from 'some/module.js';",
               Style);
  verifyFormat("import {\n"
               "  A,\n"
               "  A,\n"
               "} from 'some/module.js';",
               Style);
  verifyFormat("export {\n"
               "  A,\n"
               "  A,\n"
               "} from 'some/module.js';",
               Style);
}

TEST_F(FormatTestJS, TemplateStrings) {
  // Keeps any whitespace/indentation within the template string.
  verifyFormat("var x = `hello\n"
            "     ${name}\n"
            "  !`;",
            "var x    =    `hello\n"
                   "     ${  name    }\n"
                   "  !`;");

  verifyFormat("var x =\n"
               "    `hello ${world}` >= some();",
               getGoogleJSStyleWithColumns(34)); // Barely doesn't fit.
  verifyFormat("var x = `hello ${world}` >= some();",
               getGoogleJSStyleWithColumns(35)); // Barely fits.
  verifyFormat("var x = `hellö ${wörld}` >= söme();",
               getGoogleJSStyleWithColumns(35)); // Fits due to UTF-8.
  verifyFormat("var x = `hello\n"
            "  ${world}` >=\n"
            "    some();",
            "var x =\n"
                   "    `hello\n"
                   "  ${world}` >= some();",
                   getGoogleJSStyleWithColumns(21)); // Barely doesn't fit.
  verifyFormat("var x = `hello\n"
            "  ${world}` >= some();",
            "var x =\n"
                   "    `hello\n"
                   "  ${world}` >= some();",
                   getGoogleJSStyleWithColumns(22)); // Barely fits.

  verifyFormat("var x =\n"
               "    `h`;",
               getGoogleJSStyleWithColumns(11));
  verifyFormat("var x =\n    `multi\n  line`;", "var x = `multi\n  line`;",
               getGoogleJSStyleWithColumns(13));
  verifyFormat("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa(\n"
               "    `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`);");
  // Repro for an obscure width-miscounting issue with template strings.
  verifyFormat(
      "someLongVariable =\n"
      "    "
      "`${logPrefix[11]}/${logPrefix[12]}/${logPrefix[13]}${logPrefix[14]}`;",
      "someLongVariable = "
      "`${logPrefix[11]}/${logPrefix[12]}/${logPrefix[13]}${logPrefix[14]}`;");

  // Make sure template strings get a proper ColumnWidth assigned, even if they
  // are first token in line.
  verifyFormat(
      "var a = aaaaaaaaaaaaaaaaaaaaaaaaaaaa ||\n"
      "    `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`;");

  // Two template strings.
  verifyFormat("var x = `hello` == `hello`;");

  // Comments in template strings.
  verifyFormat("var x = `//a`;\n"
            "var y;",
            "var x =\n `//a`;\n"
                   "var y  ;");
  verifyFormat("var x = `/*a`;\n"
               "var y;",
               "var x =\n `/*a`;\n"
               "var y;");
  // Unterminated string literals in a template string.
  verifyFormat("var x = `'`;  // comment with matching quote '\n"
               "var y;");
  verifyFormat("var x = `\"`;  // comment with matching quote \"\n"
               "var y;");
  verifyFormat("it(`'aaaaaaaaaaaaaaa   `, aaaaaaaaa);",
               "it(`'aaaaaaaaaaaaaaa   `,   aaaaaaaaa) ;",
               getGoogleJSStyleWithColumns(40));
  // Backticks in a comment - not a template string.
  verifyFormat("var x = 1  // `/*a`;\n"
               "    ;",
               "var x =\n 1  // `/*a`;\n"
               "    ;");
  verifyFormat("/* ` */ var x = 1; /* ` */", "/* ` */ var x\n= 1; /* ` */");
  // Comment spans multiple template strings.
  verifyFormat("var x = `/*a`;\n"
               "var y = ` */ `;",
               "var x =\n `/*a`;\n"
               "var y =\n ` */ `;");
  // Escaped backtick.
  verifyFormat("var x = ` \\` a`;\n"
               "var y;",
               "var x = ` \\` a`;\n"
               "var y;");
  // Escaped dollar.
  verifyFormat("var x = ` \\${foo}`;\n");
}

TEST_F(FormatTestJS, NestedTemplateStrings) {
  verifyFormat(
      "var x = `<ul>${xs.map(x => `<li>${x}</li>`).join('\\n')}</ul>`;");
  verifyFormat("var x = `he${({text: 'll'}.text)}o`;");
}

TEST_F(FormatTestJS, TaggedTemplateStrings) {
  verifyFormat("var x = html`<ul>`;");
}

TEST_F(FormatTestJS, CastSyntax) {
  verifyFormat("var x = <type>foo;");
  verifyFormat("var x = foo as type;");
  verifyFormat("let x = (a + b) as\n"
               "    LongTypeIsLong;",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("foo = <Bar[]>[\n"
               "  1,  //\n"
               "  2\n"
               "];");
  verifyFormat("var x = [{x: 1} as type];");
  verifyFormat("x = x as [a, b];");
  verifyFormat("x = x as {a: string};");
  verifyFormat("x = x as (string);");
  verifyFormat("x = x! as (string);");
}

TEST_F(FormatTestJS, TypeArguments) {
  verifyFormat("class X<Y> {}");
  verifyFormat("new X<Y>();");
  verifyFormat("foo<Y>(a);");
  verifyFormat("var x: X<Y>[];");
  verifyFormat("class C extends D<E> implements F<G>, H<I> {}");
  verifyFormat("function f(a: List<any> = null) {}");
  verifyFormat("function f(): List<any> {}");
  verifyFormat("function aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa():\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb {}");
  verifyFormat("function aaaaaaaaaa(\n"
               "    aaaaaaaaaaaaaaaa: aaaaaaaaaaaaaaaaaaa,\n"
               "    aaaaaaaaaaaaaaaa: aaaaaaaaaaaaaaaaaaa):\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa {}");
}

TEST_F(FormatTestJS, UserDefinedTypeGuards) {
  verifyFormat(
      "function foo(check: Object):\n"
      "    check is {foo: string, bar: string, baz: string, foobar: string} {\n"
      "  return 'bar' in check;\n"
      "}\n");
}

TEST_F(FormatTestJS, OptionalTypes) {
  verifyFormat("function x(a?: b, c?, d?) {}");
  verifyFormat("class X {\n"
               "  y?: z;\n"
               "  z?;\n"
               "}");
  verifyFormat("interface X {\n"
               "  y?(): z;\n"
               "}");
  verifyFormat("constructor({aa}: {\n"
               "  aa?: string,\n"
               "  aaaaaaaa?: string,\n"
               "  aaaaaaaaaaaaaaa?: boolean,\n"
               "  aaaaaa?: List<string>\n"
               "}) {}");
}

TEST_F(FormatTestJS, IndexSignature) {
  verifyFormat("var x: {[k: string]: v};");
}

TEST_F(FormatTestJS, WrapAfterParen) {
  verifyFormat("xxxxxxxxxxx(\n"
               "    aaa, aaa);",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("xxxxxxxxxxx(\n"
               "    aaa, aaa, aaa,\n"
               "    aaa, aaa, aaa);",
               getGoogleJSStyleWithColumns(20));
  verifyFormat("xxxxxxxxxxx(\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaa,\n"
               "    function(x) {\n"
               "      y();  //\n"
               "    });",
               getGoogleJSStyleWithColumns(40));
  verifyFormat("while (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa &&\n"
               "       bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb) {\n}");
}

TEST_F(FormatTestJS, JSDocAnnotations) {
  verifyFormat("/**\n"
               " * @export {this.is.a.long.path.to.a.Type}\n"
               " */",
               "/**\n"
               " * @export {this.is.a.long.path.to.a.Type}\n"
               " */",
               getGoogleJSStyleWithColumns(20));
}

TEST_F(FormatTestJS, RequoteStringsSingle) {
  verifyFormat("var x = 'foo';", "var x = \"foo\";");
  verifyFormat("var x = 'fo\\'o\\'';", "var x = \"fo'o'\";");
  verifyFormat("var x = 'fo\\'o\\'';", "var x = \"fo\\'o'\";");
  verifyFormat(
      "var x =\n"
      "    'foo\\'';",
      // Code below is 15 chars wide, doesn't fit into the line with the
      // \ escape added.
      "var x = \"foo'\";", getGoogleJSStyleWithColumns(15));
  // Removes no-longer needed \ escape from ".
  verifyFormat("var x = 'fo\"o';", "var x = \"fo\\\"o\";");
  // Code below fits into 15 chars *after* removing the \ escape.
  verifyFormat("var x = 'fo\"o';", "var x = \"fo\\\"o\";",
               getGoogleJSStyleWithColumns(15));
  verifyFormat("// clang-format off\n"
               "let x = \"double\";\n"
               "// clang-format on\n"
               "let x = 'single';\n",
               "// clang-format off\n"
               "let x = \"double\";\n"
               "// clang-format on\n"
               "let x = \"single\";\n");
}

TEST_F(FormatTestJS, RequoteAndIndent) {
  verifyFormat("let x = someVeryLongFunctionThatGoesOnAndOn(\n"
               "    'double quoted string that needs wrapping');",
               "let x = someVeryLongFunctionThatGoesOnAndOn("
               "\"double quoted string that needs wrapping\");");

  verifyFormat("let x =\n"
               "    'foo\\'oo';\n"
               "let x =\n"
               "    'foo\\'oo';",
               "let x=\"foo'oo\";\n"
               "let x=\"foo'oo\";",
               getGoogleJSStyleWithColumns(15));
}

TEST_F(FormatTestJS, RequoteStringsDouble) {
  FormatStyle DoubleQuotes = getGoogleStyle(FormatStyle::LK_JavaScript);
  DoubleQuotes.JavaScriptQuotes = FormatStyle::JSQS_Double;
  verifyFormat("var x = \"foo\";", DoubleQuotes);
  verifyFormat("var x = \"foo\";", "var x = 'foo';", DoubleQuotes);
  verifyFormat("var x = \"fo'o\";", "var x = 'fo\\'o';", DoubleQuotes);
}

TEST_F(FormatTestJS, RequoteStringsLeave) {
  FormatStyle LeaveQuotes = getGoogleStyle(FormatStyle::LK_JavaScript);
  LeaveQuotes.JavaScriptQuotes = FormatStyle::JSQS_Leave;
  verifyFormat("var x = \"foo\";", LeaveQuotes);
  verifyFormat("var x = 'foo';", LeaveQuotes);
}

TEST_F(FormatTestJS, SupportShebangLines) {
  verifyFormat("#!/usr/bin/env node\n"
               "var x = hello();",
               "#!/usr/bin/env node\n"
               "var x   =  hello();");
}

TEST_F(FormatTestJS, NonNullAssertionOperator) {
  verifyFormat("let x = foo!.bar();\n");
  verifyFormat("let x = foo ? bar! : baz;\n");
  verifyFormat("let x = !foo;\n");
  verifyFormat("let x = foo[0]!;\n");
  verifyFormat("let x = (foo)!;\n");
  verifyFormat("let x = {foo: 1}!;\n");
}

TEST_F(FormatTestJS, Conditional) {
  verifyFormat("y = x ? 1 : 2;");
  verifyFormat("x ? 1 : 2;");
  verifyFormat("class Foo {\n"
               "  field = true ? 1 : 2;\n"
               "  method(a = true ? 1 : 2) {}\n"
               "}");
}

} // end namespace tooling
} // end namespace clang
