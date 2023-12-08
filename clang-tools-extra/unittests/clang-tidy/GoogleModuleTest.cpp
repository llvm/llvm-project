#include "ClangTidyTest.h"
#include "google/ExplicitConstructorCheck.h"
#include "google/GlobalNamesInHeadersCheck.h"
#include "gtest/gtest.h"

using namespace clang::tidy::google;

namespace clang {
namespace tidy {
namespace test {

TEST(ExplicitConstructorCheckTest, SingleArgumentConstructorsOnly) {
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(int i, int j); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(const C&); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck, "class C { C(C&&); };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck,
                    "class C { C(const C&) = delete; };");
  EXPECT_NO_CHANGES(ExplicitConstructorCheck,
                    "class C { C(int) = delete; };");
}

TEST(ExplicitConstructorCheckTest, Basic) {
  EXPECT_EQ("class C { explicit C(int i); };",
            runCheckOnCode<ExplicitConstructorCheck>("class C { C(int i); };"));
}

TEST(ExplicitConstructorCheckTest, DefaultParameters) {
  EXPECT_EQ("class C { explicit C(int i, int j = 0); };",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class C { C(int i, int j = 0); };"));
}

TEST(ExplicitConstructorCheckTest, OutOfLineDefinitions) {
  EXPECT_EQ("class C { explicit C(int i); }; C::C(int i) {}",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class C { C(int i); }; C::C(int i) {}"));
}

TEST(ExplicitConstructorCheckTest, RemoveExplicit) {
  EXPECT_EQ("class A { A(const A&); };\n"
            "class B { /*asdf*/  B(B&&); };\n"
            "class C { /*asdf*/  C(const C&, int i = 0); };",
            runCheckOnCode<ExplicitConstructorCheck>(
                "class A { explicit    A(const A&); };\n"
                "class B { explicit   /*asdf*/  B(B&&); };\n"
                "class C { explicit/*asdf*/  C(const C&, int i = 0); };"));
}

TEST(ExplicitConstructorCheckTest, RemoveExplicitWithMacros) {
  EXPECT_EQ(
      "#define A(T) class T##Bar { explicit T##Bar(const T##Bar &b) {} };\n"
      "A(Foo);",
      runCheckOnCode<ExplicitConstructorCheck>(
          "#define A(T) class T##Bar { explicit T##Bar(const T##Bar &b) {} };\n"
          "A(Foo);"));
}

class GlobalNamesInHeadersCheckTest : public ::testing::Test {
protected:
  bool runCheckOnCode(const std::string &Code, const std::string &Filename) {
    static const char *const Header = "namespace std {\n"
                                      "class string {};\n"
                                      "}  // namespace std\n"
                                      "\n"
                                      "#define SOME_MACRO(x) using x\n";
    std::vector<ClangTidyError> Errors;
    std::vector<std::string> Args;
    if (!StringRef(Filename).endswith(".cpp")) {
      Args.emplace_back("-xc++-header");
    }
    test::runCheckOnCode<readability::GlobalNamesInHeadersCheck>(
        Header + Code, &Errors, Filename, Args);
    if (Errors.empty())
      return false;
    assert(Errors.size() == 1);
    assert(
        Errors[0].Message.Message ==
        "using declarations in the global namespace in headers are prohibited");
    return true;
  }
};

TEST_F(GlobalNamesInHeadersCheckTest, UsingDeclarations) {
  EXPECT_TRUE(runCheckOnCode("using std::string;", "foo.h"));
  EXPECT_FALSE(runCheckOnCode("using std::string;", "foo.cpp"));
  EXPECT_FALSE(runCheckOnCode("namespace my_namespace {\n"
                              "using std::string;\n"
                              "}  // my_namespace\n",
                              "foo.h"));
  EXPECT_FALSE(runCheckOnCode("SOME_MACRO(std::string);", "foo.h"));
}

TEST_F(GlobalNamesInHeadersCheckTest, UsingDirectives) {
  EXPECT_TRUE(runCheckOnCode("using namespace std;", "foo.h"));
  EXPECT_FALSE(runCheckOnCode("using namespace std;", "foo.cpp"));
  EXPECT_FALSE(runCheckOnCode("namespace my_namespace {\n"
                              "using namespace std;\n"
                              "}  // my_namespace\n",
                              "foo.h"));
  EXPECT_FALSE(runCheckOnCode("SOME_MACRO(namespace std);", "foo.h"));
}

TEST_F(GlobalNamesInHeadersCheckTest, RegressionAnonymousNamespace) {
  EXPECT_FALSE(runCheckOnCode("namespace {}", "foo.h"));
}

} // namespace test
} // namespace tidy
} // namespace clang
