//===- unittest/Format/FormatTestJava.cpp - Formatting tests for Java -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatTestBase.h"

#define DEBUG_TYPE "format-test"

namespace clang {
namespace format {
namespace test {
namespace {

class FormatTestJava : public test::FormatTestBase {
protected:
  FormatStyle getDefaultStyle() const override {
    return getGoogleStyle(FormatStyle::LK_Java);
  }

  static FormatStyle getStyleWithColumns(unsigned ColumnLimit) {
    FormatStyle Style = getGoogleStyle(FormatStyle::LK_Java);
    Style.ColumnLimit = ColumnLimit;
    return Style;
  }
};

TEST_F(FormatTestJava, NoAlternativeOperatorNames) {
  verifyFormat("someObject.and();");
}

TEST_F(FormatTestJava, UnderstandsCasts) {
  verifyFormat("a[b >> 1] = (byte) (c() << 4);");
}

TEST_F(FormatTestJava, FormatsInstanceOfLikeOperators) {
  FormatStyle Style = getStyleWithColumns(50);
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
               "    instanceof bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
  Style.BreakBeforeBinaryOperators = FormatStyle::BOS_None;
  verifyFormat("return aaaaaaaaaaaaaaaaaaaaaaaaaaaaa instanceof\n"
               "    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb;",
               Style);
  verifyFormat("return aaaaaaaaaaaaaaaaaaa instanceof bbbbbbbbbbbbbbbbbbbbbbb\n"
               "    && ccccccccccccccccccc instanceof dddddddddddddddddddddd;");
}

TEST_F(FormatTestJava, Chromium) {
  verifyFormat("class SomeClass {\n"
               "    void f() {}\n"
               "    int g() {\n"
               "        return 0;\n"
               "    }\n"
               "    void h() {\n"
               "        while (true) f();\n"
               "        for (;;) f();\n"
               "        if (true) f();\n"
               "    }\n"
               "}",
               getChromiumStyle(FormatStyle::LK_Java));
}

TEST_F(FormatTestJava, QualifiedNames) {
  verifyFormat("public some.package.Type someFunction( // comment\n"
               "    int parameter) {}");
}

TEST_F(FormatTestJava, ClassKeyword) {
  verifyFormat("SomeClass.class.getName();");
  verifyFormat("Class c = SomeClass.class;");
}

TEST_F(FormatTestJava, ClassDeclarations) {
  verifyFormat("public class SomeClass {\n"
               "  private int a;\n"
               "  private int b;\n"
               "}");
  verifyFormat("public class A {\n"
               "  class B {\n"
               "    int i;\n"
               "  }\n"
               "  class C {\n"
               "    int j;\n"
               "  }\n"
               "}");
  verifyFormat("public class A extends B.C {}");

  verifyFormat("abstract class SomeClass\n"
               "    extends SomeOtherClass implements SomeInterface {}",
               getStyleWithColumns(60));
  verifyFormat("abstract class SomeClass extends SomeOtherClass\n"
               "    implements SomeInterfaceeeeeeeeeeeee {}",
               getStyleWithColumns(60));
  verifyFormat("abstract class SomeClass\n"
               "    extends SomeOtherClass\n"
               "    implements SomeInterface {}",
               getStyleWithColumns(40));
  verifyFormat("abstract class SomeClass\n"
               "    extends SomeOtherClass\n"
               "    implements SomeInterface,\n"
               "               AnotherInterface {}",
               getStyleWithColumns(40));
  verifyFormat("abstract class SomeClass\n"
               "    implements SomeInterface, AnotherInterface {}",
               getStyleWithColumns(60));
  verifyFormat("@SomeAnnotation()\n"
               "abstract class aaaaaaaaaaaa\n"
               "    extends bbbbbbbbbbbbbbb implements cccccccccccc {}",
               getStyleWithColumns(76));
  verifyFormat("@SomeAnnotation()\n"
               "abstract class aaaaaaaaa<a>\n"
               "    extends bbbbbbbbbbbb<b> implements cccccccccccc {}",
               getStyleWithColumns(76));
  verifyFormat("interface SomeInterface<A> extends Foo, Bar {\n"
               "  void doStuff(int theStuff);\n"
               "  void doMoreStuff(int moreStuff);\n"
               "}");
  verifyFormat("public interface SomeInterface {\n"
               "  void doStuff(int theStuff);\n"
               "  void doMoreStuff(int moreStuff);\n"
               "  default void doStuffWithDefault() {}\n"
               "}");
  verifyFormat("@interface SomeInterface {\n"
               "  void doStuff(int theStuff);\n"
               "  void doMoreStuff(int moreStuff);\n"
               "}");
  verifyFormat("public @interface SomeInterface {\n"
               "  void doStuff(int theStuff);\n"
               "  void doMoreStuff(int moreStuff);\n"
               "}");
  verifyFormat("class A {\n"
               "  public @interface SomeInterface {\n"
               "    int stuff;\n"
               "    void doMoreStuff(int moreStuff);\n"
               "  }\n"
               "}");
  verifyFormat("class A {\n"
               "  public @interface SomeInterface {}\n"
               "}");
}

TEST_F(FormatTestJava, AnonymousClasses) {
  verifyFormat("return new A() {\n"
               "  public String toString() {\n"
               "    return \"NotReallyA\";\n"
               "  }\n"
               "};");
  verifyFormat("A a = new A() {\n"
               "  public String toString() {\n"
               "    return \"NotReallyA\";\n"
               "  }\n"
               "};");
}

TEST_F(FormatTestJava, EnumDeclarations) {
  verifyFormat("enum SomeThing { ABC, CDE }");
  // A C++ keyword should not mess things up.
  verifyFormat("enum union { ABC, CDE }");
  verifyFormat("enum SomeThing {\n"
               "  ABC,\n"
               "  CDE,\n"
               "}");
  verifyFormat("public class SomeClass {\n"
               "  enum SomeThing { ABC, CDE }\n"
               "  void f() {}\n"
               "}");
  verifyFormat("public class SomeClass implements SomeInterface {\n"
               "  enum SomeThing { ABC, CDE }\n"
               "  void f() {}\n"
               "}");
  verifyFormat("enum SomeThing {\n"
               "  ABC,\n"
               "  CDE;\n"
               "  void f() {}\n"
               "}");
  verifyFormat("enum SomeThing {\n"
               "  void f() {}");
  verifyFormat("enum SomeThing {\n"
               "  ABC(1, \"ABC\"),\n"
               "  CDE(2, \"CDE\");\n"
               "  Something(int i, String s) {}\n"
               "}");
  verifyFormat("enum SomeThing {\n"
               "  ABC(new int[] {1, 2}),\n"
               "  CDE(new int[] {2, 3});\n"
               "  Something(int[] i) {}\n"
               "}");
  verifyFormat("public enum SomeThing {\n"
               "  ABC {\n"
               "    public String toString() {\n"
               "      return \"ABC\";\n"
               "    }\n"
               "  },\n"
               "  CDE {\n"
               "    @Override\n"
               "    public String toString() {\n"
               "      return \"CDE\";\n"
               "    }\n"
               "  };\n"
               "  public void f() {}\n"
               "}");
  verifyFormat("private enum SomeEnum implements Foo<?, B> {\n"
               "  ABC {\n"
               "    @Override\n"
               "    public String toString() {\n"
               "      return \"ABC\";\n"
               "    }\n"
               "  },\n"
               "  CDE {\n"
               "    @Override\n"
               "    public String toString() {\n"
               "      return \"CDE\";\n"
               "    }\n"
               "  };\n"
               "}");
  verifyFormat("public enum VeryLongEnum {\n"
               "  ENUM_WITH_MANY_PARAMETERS(\n"
               "      \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaa\", \"bbbbbbbbbbbbbbbb\", "
               "\"cccccccccccccccccccccccc\"),\n"
               "  SECOND_ENUM(\"a\", \"b\", \"c\");\n"
               "  private VeryLongEnum(String a, String b, String c) {}\n"
               "}");
}

TEST_F(FormatTestJava, ArrayInitializers) {
  verifyFormat("new int[] {1, 2, 3, 4};");
  verifyFormat("new int[] {\n"
               "    1,\n"
               "    2,\n"
               "    3,\n"
               "    4,\n"
               "};");

  FormatStyle Style = getStyleWithColumns(65);
  Style.Cpp11BracedListStyle = FormatStyle::BLS_Block;
  verifyFormat(
      "expected = new int[] { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,\n"
      "  100, 100, 100, 100, 100, 100, 100, 100, 100, 100 };",
      Style);
}

TEST_F(FormatTestJava, ThrowsDeclarations) {
  verifyFormat("public void doSooooooooooooooooooooooooooomething()\n"
               "    throws LooooooooooooooooooooooooooooongException {}");
  verifyFormat("public void doSooooooooooooooooooooooooooomething()\n"
               "    throws LoooooooooongException, LooooooooooongException {}");
}

TEST_F(FormatTestJava, Annotations) {
  verifyFormat("@Override\n"
               "public String toString() {}");
  verifyFormat("@Override\n"
               "@Nullable\n"
               "public String getNameIfPresent() {}");
  verifyFormat("@Override // comment\n"
               "@Nullable\n"
               "public String getNameIfPresent() {}");
  verifyFormat("@java.lang.Override // comment\n"
               "@Nullable\n"
               "public String getNameIfPresent() {}");

  verifyFormat("@SuppressWarnings(value = \"unchecked\")\n"
               "public void doSomething() {}");
  verifyFormat("@SuppressWarnings(value = \"unchecked\")\n"
               "@Author(name = \"abc\")\n"
               "public void doSomething() {}");

  verifyFormat("DoSomething(new A() {\n"
               "  @Override\n"
               "  public String toString() {}\n"
               "});");

  verifyFormat("void SomeFunction(@Nullable String something) {}");
  verifyFormat("void SomeFunction(@org.llvm.Nullable String something) {}");

  verifyFormat("@Partial @Mock DataLoader loader;");
  verifyFormat("@Partial\n"
               "@Mock\n"
               "DataLoader loader;",
               getChromiumStyle(FormatStyle::LK_Java));
  verifyFormat("@SuppressWarnings(value = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\")\n"
               "public static int iiiiiiiiiiiiiiiiiiiiiiii;");

  verifyFormat("@SomeAnnotation(\"With some really looooooooooooooong text\")\n"
               "private static final long something = 0L;");
  verifyFormat("@org.llvm.Qualified(\"With some really looooooooooong text\")\n"
               "private static final long something = 0L;");
  verifyFormat("@Mock\n"
               "DataLoader loooooooooooooooooooooooader =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               getStyleWithColumns(60));
  verifyFormat("@org.llvm.QualifiedMock\n"
               "DataLoader loooooooooooooooooooooooader =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa;",
               getStyleWithColumns(60));
  verifyFormat("@Test(a)\n"
               "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa =\n"
               "    aaaaaaaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaaaa);");
  verifyFormat("@SomeAnnotation(\n"
               "    aaaaaaaaaaaaaaaaa, aaaaaaaaaaaaaaaaaaaa)\n"
               "int i;",
               getStyleWithColumns(50));
  verifyFormat("@Test\n"
               "ReturnType doSomething(\n"
               "    String aaaaaaaaaaaaa, String bbbbbbbbbbbbbbb) {}",
               getStyleWithColumns(60));
  verifyFormat("{\n"
               "  boolean someFunction(\n"
               "      @Param(aaaaaaaaaaaaaaaa) String aaaaa,\n"
               "      String bbbbbbbbbbbbbbb) {}\n"
               "}",
               getStyleWithColumns(60));
  verifyFormat("@Annotation(\"Some\"\n"
               "    + \" text\")\n"
               "List<Integer> list;");

  verifyFormat(
      "@Test\n"
      "@Feature({\"Android-TabSwitcher\"})\n"
      "@CommandLineFlags.Add({ChromeSwitches.DISABLE_FIRST_RUN_EXPERIENCE})\n"
      "@Features.EnableFeatures({FEATURE})\n"
      "public void test(@Foo.bar(\"baz\") @Quux.Qoob int theFirstParaaaaam,\n"
      "    @Foo.bar(\"baz\") @Quux.Qoob int theSecondParaaaaaaaaaaaaaaaam) {}");
}

TEST_F(FormatTestJava, Generics) {
  verifyFormat("Iterable<?> a;");
  verifyFormat("Iterable<?> a;");
  verifyFormat("Iterable<? extends SomeObject> a;");

  verifyFormat("A.<B>doSomething();");
  verifyFormat("A.<B<C>>doSomething();");
  verifyFormat("A.<B<C<D>>>doSomething();");
  verifyFormat("A.<B<C<D<E>>>>doSomething();");

  verifyFormat("OrderedPair<String, List<Box<Integer>>> p = null;");

  verifyFormat("@Override\n"
               "public Map<String, ?> getAll() {}");

  verifyFormat("public <R> ArrayList<R> get() {}");
  verifyFormat("protected <R> ArrayList<R> get() {}");
  verifyFormat("private <R> ArrayList<R> get() {}");
  verifyFormat("public static <R> ArrayList<R> get() {}");
  verifyFormat("public static native <R> ArrayList<R> get();");
  verifyFormat("public final <X> Foo foo() {}");
  verifyFormat("public abstract <X> Foo foo();");
  verifyFormat("<T extends B> T getInstance(Class<T> type);");
  verifyFormat("Function<F, ? extends T> function;");

  verifyFormat("private Foo<X, Y>[] foos;");
  verifyFormat("Foo<X, Y>[] foos = this.foos;");
  verifyFormat("return (a instanceof List<?>)\n"
               "    ? aaaaaaaaaaaaaaaaaaaaaaa(aaaaaaaaaaaaaaaaaaaaa)\n"
               "    : aaaaaaaaaaaaaaaaaaaaaaa;",
               getStyleWithColumns(60));

  verifyFormat(
      "SomeLoooooooooooooooooooooongType name =\n"
      "    SomeType.foo(someArgument)\n"
      "        .<X>method()\n"
      "        .aaaaaaaaaaaaaaaaaaa()\n"
      "        .aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa();");
}

TEST_F(FormatTestJava, StringConcatenation) {
  verifyFormat("String someString = \"abc\"\n"
               "    + \"cde\";");
}

TEST_F(FormatTestJava, TryCatchFinally) {
  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException e) {\n"
               "  HandleException(e);\n"
               "}");
  verifyFormat("try {\n"
               "  Something();\n"
               "} finally {\n"
               "  AlwaysDoThis();\n"
               "}");
  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException e) {\n"
               "  HandleException(e);\n"
               "} finally {\n"
               "  AlwaysDoThis();\n"
               "}");

  verifyFormat("try {\n"
               "  Something();\n"
               "} catch (SomeException | OtherException e) {\n"
               "  HandleException(e);\n"
               "}");
}

TEST_F(FormatTestJava, TryWithResources) {
  verifyFormat("try (SomeResource rs = someFunction()) {\n"
               "  Something();\n"
               "}");
  verifyFormat("try (SomeResource rs = someFunction()) {\n"
               "  Something();\n"
               "} catch (SomeException e) {\n"
               "  HandleException(e);\n"
               "}");
}

TEST_F(FormatTestJava, SynchronizedKeyword) {
  verifyFormat("synchronized (mData) {\n"
               "  // ...\n"
               "}");

  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Java);
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;

  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  Style.BraceWrapping.AfterFunction = false;
  verifyFormat("synchronized (mData)\n"
               "{\n"
               "  // ...\n"
               "}",
               Style);

  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Never;
  Style.BraceWrapping.AfterFunction = true;
  verifyFormat("synchronized (mData) {\n"
               "  // ...\n"
               "}",
               Style);
}

TEST_F(FormatTestJava, AssertKeyword) {
  verifyFormat("assert a && b;");
  verifyFormat("assert (a && b);");
}

TEST_F(FormatTestJava, PackageDeclarations) {
  verifyFormat("package some.really.loooooooooooooooooooooong.package;",
               getStyleWithColumns(50));
}

TEST_F(FormatTestJava, ImportDeclarations) {
  verifyFormat("import some.really.loooooooooooooooooooooong.imported.Class;",
               getStyleWithColumns(50));
  verifyFormat("import static some.really.looooooooooooooooong.imported.Class;",
               getStyleWithColumns(50));
}

TEST_F(FormatTestJava, MethodDeclarations) {
  verifyFormat("void methodName(Object arg1,\n"
               "    Object arg2, Object arg3) {}",
               getStyleWithColumns(40));
  verifyFormat("void methodName(\n"
               "    Object arg1, Object arg2) {}",
               getStyleWithColumns(40));
}

TEST_F(FormatTestJava, MethodReference) {
  EXPECT_EQ("private void foo() {\n"
            "  f(this::methodReference);\n"
            "  f(C.super::methodReference);\n"
            "  Consumer<String> c = System.out::println;\n"
            "  Iface<Integer> mRef = Ty::<Integer>meth;\n"
            "}",
            format("private void foo() {\n"
                   "  f(this ::methodReference);\n"
                   "  f(C.super ::methodReference);\n"
                   "  Consumer<String> c = System.out ::println;\n"
                   "  Iface<Integer> mRef = Ty :: <Integer> meth;\n"
                   "}"));
}

TEST_F(FormatTestJava, CppKeywords) {
  verifyFormat("public void union(Type a, Type b);");
  verifyFormat("public void struct(Object o);");
  verifyFormat("public void delete(Object o);");
  verifyFormat("return operator && (aa);");
}

TEST_F(FormatTestJava, NeverAlignAfterReturn) {
  verifyFormat("return aaaaaaaaaaaaaaaaaaa\n"
               "    && bbbbbbbbbbbbbbbbbbb\n"
               "    && ccccccccccccccccccc;",
               getStyleWithColumns(40));
  verifyFormat("return (result == null)\n"
               "    ? aaaaaaaaaaaaaaaaa\n"
               "    : bbbbbbbbbbbbbbbbb;",
               getStyleWithColumns(40));
  verifyFormat("return aaaaaaaaaaaaaaaaaaa()\n"
               "    .bbbbbbbbbbbbbbbbbbb()\n"
               "    .ccccccccccccccccccc();",
               getStyleWithColumns(40));
  verifyFormat("return aaaaaaaaaaaaaaaaaaa()\n"
               "    .bbbbbbbbbbbbbbbbbbb(\n"
               "        ccccccccccccccc)\n"
               "    .ccccccccccccccccccc();",
               getStyleWithColumns(40));
}

TEST_F(FormatTestJava, FormatsInnerBlocks) {
  verifyFormat("someObject.someFunction(new Runnable() {\n"
               "  @Override\n"
               "  public void run() {\n"
               "    System.out.println(42);\n"
               "  }\n"
               "}, someOtherParameter);");
  verifyFormat("someFunction(new Runnable() {\n"
               "  public void run() {\n"
               "    System.out.println(42);\n"
               "  }\n"
               "});");
  verifyFormat("someObject.someFunction(\n"
               "    new Runnable() {\n"
               "      @Override\n"
               "      public void run() {\n"
               "        System.out.println(42);\n"
               "      }\n"
               "    },\n"
               "    new Runnable() {\n"
               "      @Override\n"
               "      public void run() {\n"
               "        System.out.println(43);\n"
               "      }\n"
               "    },\n"
               "    someOtherParameter);");
}

TEST_F(FormatTestJava, FormatsLambdas) {
  verifyFormat("(aaaaaaaaaa, bbbbbbbbbb) -> aaaaaaaaaa + bbbbbbbbbb;");
  verifyFormat("(aaaaaaaaaa, bbbbbbbbbb)\n"
               "    -> aaaaaaaaaa + bbbbbbbbbb;",
               getStyleWithColumns(40));
  verifyFormat("Runnable someLambda = () -> DoSomething();");
  verifyFormat("Runnable someLambda = () -> {\n"
               "  DoSomething();\n"
               "}");

  verifyFormat("Runnable someLambda =\n"
               "    (int aaaaa) -> DoSomething(aaaaa);",
               getStyleWithColumns(40));
}

TEST_F(FormatTestJava, BreaksStringLiterals) {
  verifyFormat("x = \"some text \"\n"
               "    + \"other\";",
               "x = \"some text other\";", getStyleWithColumns(18));
}

TEST_F(FormatTestJava, AlignsBlockComments) {
  EXPECT_EQ("/*\n"
            " * Really multi-line\n"
            " * comment.\n"
            " */\n"
            "void f() {}",
            format("  /*\n"
                   "   * Really multi-line\n"
                   "   * comment.\n"
                   "   */\n"
                   "  void f() {}"));
}

TEST_F(FormatTestJava, AlignDeclarations) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Java);
  Style.AlignConsecutiveDeclarations.Enabled = true;
  verifyFormat("private final String[]       args;\n"
               "private final A_ParserHelper parserHelper;\n"
               "private final int            numOfCmdArgs;\n"
               "private int                  numOfCmdArgs;\n"
               "private String[]             args;",
               Style);
}

TEST_F(FormatTestJava, KeepsDelimitersOnOwnLineInJavaDocComments) {
  EXPECT_EQ("/**\n"
            " * javadoc line 1\n"
            " * javadoc line 2\n"
            " */",
            format("/** javadoc line 1\n"
                   " * javadoc line 2 */"));
}

TEST_F(FormatTestJava, RetainsLogicalShifts) {
  verifyFormat("void f() {\n"
               "  int a = 1;\n"
               "  a >>>= 1;\n"
               "}");
  verifyFormat("void f() {\n"
               "  int a = 1;\n"
               "  a = a >>> 1;\n"
               "}");
}

TEST_F(FormatTestJava, ShortFunctions) {
  FormatStyle Style = getLLVMStyle(FormatStyle::LK_Java);
  Style.AllowShortFunctionsOnASingleLine = FormatStyle::SFS_Inline;
  verifyFormat("enum Enum {\n"
               "  E1,\n"
               "  E2;\n"
               "  void f() { return; }\n"
               "}",
               Style);
}

TEST_F(FormatTestJava, ConfigurableSpacesInSquareBrackets) {
  FormatStyle Spaces = getLLVMStyle(FormatStyle::LK_Java);

  verifyFormat("Object[] arguments", Spaces);
  verifyFormat("final Class<?>[] types = new Class<?>[numElements];", Spaces);
  verifyFormat("types[i] = arguments[i].getClass();", Spaces);

  Spaces.SpacesInSquareBrackets = true;

  verifyFormat("Object[ ] arguments", Spaces);
  verifyFormat("final Class<?>[ ] types = new Class<?>[ numElements ];",
               Spaces);
  verifyFormat("types[ i ] = arguments[ i ].getClass();", Spaces);
}

TEST_F(FormatTestJava, SwitchExpression) {
  auto Style = getLLVMStyle(FormatStyle::LK_Java);
  EXPECT_TRUE(Style.AllowShortCaseExpressionOnASingleLine);

  verifyFormat("foo(switch (day) {\n"
               "  case THURSDAY, SATURDAY -> 8;\n"
               "  case WEDNESDAY -> 9;\n"
               "  default -> 1;\n"
               "});",
               Style);

  constexpr StringRef Code1("i = switch (day) {\n"
                            "  case THURSDAY, SATURDAY -> 8;\n"
                            "  case WEDNESDAY -> 9;\n"
                            "  default -> 0;\n"
                            "};");
  verifyFormat(Code1, Style);

  Style.IndentCaseLabels = true;
  verifyFormat(Code1, Style);

  constexpr StringRef Code2("i = switch (day) {\n"
                            "  case THURSDAY, SATURDAY -> {\n"
                            "    foo();\n"
                            "    yield 8;\n"
                            "  }\n"
                            "  case WEDNESDAY -> {\n"
                            "    bar();\n"
                            "    yield 9;\n"
                            "  }\n"
                            "  default -> {\n"
                            "    yield 0;\n"
                            "  }\n"
                            "};");
  verifyFormat(Code2, Style);

  Style.IndentCaseLabels = false;
  verifyFormat(Code2, Style);

  constexpr StringRef Code3("switch (day) {\n"
                            "case THURSDAY, SATURDAY -> i = 8;\n"
                            "case WEDNESDAY -> i = 9;\n"
                            "default -> i = 0;\n"
                            "};");
  verifyFormat(Code3, Style);

  Style.IndentCaseLabels = true;
  verifyFormat("switch (day) {\n"
               "  case THURSDAY, SATURDAY -> i = 8;\n"
               "  case WEDNESDAY -> i = 9;\n"
               "  default -> i = 0;\n"
               "};",
               Code3, Style);
}

TEST_F(FormatTestJava, ShortCaseExpression) {
  auto Style = getLLVMStyle(FormatStyle::LK_Java);

  verifyFormat("i = switch (a) {\n"
               "  case 1 -> 1;\n"
               "  case 2 -> // comment\n"
               "    2;\n"
               "  case 3 ->\n"
               "    // comment\n"
               "    3;\n"
               "  case 4 -> 4; // comment\n"
               "  default -> 0;\n"
               "};",
               Style);

  verifyNoChange("i = switch (a) {\n"
                 "  case 1 -> 1;\n"
                 "  // comment\n"
                 "  case 2 -> 2;\n"
                 "  // comment 1\n"
                 "  // comment 2\n"
                 "  case 3 -> 3; /* comment */\n"
                 "  case 4 -> /* comment */ 4;\n"
                 "  case 5 -> x + /* comment */ 1;\n"
                 "  default ->\n"
                 "    0; // comment line 1\n"
                 "       // comment line 2\n"
                 "};",
                 Style);

  Style.ColumnLimit = 18;
  verifyFormat("i = switch (a) {\n"
               "  case Monday ->\n"
               "    1;\n"
               "  default -> 9999;\n"
               "};",
               Style);

  Style.ColumnLimit = 80;
  Style.AllowShortCaseExpressionOnASingleLine = false;
  Style.IndentCaseLabels = true;
  verifyFormat("i = switch (n) {\n"
               "  default /*comments*/ ->\n"
               "    1;\n"
               "  case 0 ->\n"
               "    0;\n"
               "};",
               Style);

  Style.AllowShortCaseExpressionOnASingleLine = true;
  Style.BreakBeforeBraces = FormatStyle::BS_Custom;
  Style.BraceWrapping.AfterCaseLabel = true;
  Style.BraceWrapping.AfterControlStatement = FormatStyle::BWACS_Always;
  verifyFormat("i = switch (n)\n"
               "{\n"
               "  case 0 ->\n"
               "  {\n"
               "    yield 0;\n"
               "  }\n"
               "  default ->\n"
               "  {\n"
               "    yield 1;\n"
               "  }\n"
               "};",
               Style);
}

TEST_F(FormatTestJava, AlignCaseArrows) {
  auto Style = getLLVMStyle(FormatStyle::LK_Java);
  Style.AlignConsecutiveShortCaseStatements.Enabled = true;

  verifyFormat("foo(switch (day) {\n"
               "  case THURSDAY, SATURDAY -> 8;\n"
               "  case WEDNESDAY ->          9;\n"
               "  default ->                 1;\n"
               "});",
               Style);

  verifyFormat("i = switch (day) {\n"
               "  case THURSDAY, SATURDAY -> 8;\n"
               "  case WEDNESDAY ->          9;\n"
               "  default ->                 0;\n"
               "};",
               Style);

  verifyFormat("switch (day) {\n"
               "case THURSDAY, SATURDAY -> i = 8;\n"
               "case WEDNESDAY ->          i = 9;\n"
               "default ->                 i = 0;\n"
               "};",
               Style);

  Style.AlignConsecutiveShortCaseStatements.AlignCaseArrows = true;

  verifyFormat("foo(switch (day) {\n"
               "  case THURSDAY, SATURDAY -> 8;\n"
               "  case WEDNESDAY          -> 9;\n"
               "  default                 -> 1;\n"
               "});",
               Style);

  verifyFormat("i = switch (day) {\n"
               "  case THURSDAY, SATURDAY -> 8;\n"
               "  case WEDNESDAY          -> 9;\n"
               "  default                 -> 0;\n"
               "};",
               Style);

  verifyFormat("switch (day) {\n"
               "case THURSDAY, SATURDAY -> i = 8;\n"
               "case WEDNESDAY          -> i = 9;\n"
               "default                 -> i = 0;\n"
               "};",
               Style);
}

TEST_F(FormatTestJava, TextBlock) {
  verifyNoChange("String myStr = \"\"\"\n"
                 "hello\n"
                 "there\n"
                 "\"\"\";");

  verifyNoChange("String tb = \"\"\"\n"
                 "            the new\"\"\";");

  verifyNoChange("System.out.println(\"\"\"\n"
                 "    This is the first line\n"
                 "    This is the second line\n"
                 "    \"\"\");");

  verifyNoChange("void writeHTML() {\n"
                 "  String html = \"\"\" \n"
                 "                <html>\n"
                 "                    <p>Hello World.</p>\n"
                 "                </html>\n"
                 "\"\"\";\n"
                 "  writeOutput(html);\n"
                 "}");

  verifyNoChange("String colors = \"\"\"\t\n"
                 "    red\n"
                 "    green\n"
                 "    blue\"\"\".indent(4);");

  verifyNoChange("String code = \"\"\"\n"
                 "    String source = \\\"\"\"\n"
                 "        String message = \"Hello, World!\";\n"
                 "        System.out.println(message);\n"
                 "        \\\"\"\";\n"
                 "    \"\"\";");

  verifyNoChange(
      "class Outer {\n"
      "  void printPoetry() {\n"
      "    String lilacs = \"\"\"\n"
      "Passing the apple-tree blows of white and pink in the orchards\n"
      "\"\"\";\n"
      "    System.out.println(lilacs);\n"
      "  }\n"
      "}");

  verifyNoChange("String name = \"\"\"\r\n"
                 "        red\n"
                 "        green\n"
                 "        blue\\\n"
                 "    \"\"\";");

  verifyFormat("String name = \"\"\"Pat Q. Smith\"\"\";");

  verifyNoChange("String name = \"\"\"\n"
                 "              Pat Q. Smith");
}

} // namespace
} // namespace test
} // namespace format
} // namespace clang
