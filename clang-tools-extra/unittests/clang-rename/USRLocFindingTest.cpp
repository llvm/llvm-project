#include "USRFindingAction.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"
#include <map>
#include <set>
#include <vector>

namespace clang {
namespace rename {
namespace test {

// Determines if the symbol group invariants hold. To recap, those invariants
// are:
//  (1) All symbols in the same symbol group share the same USR.
//  (2) Two symbols from two different groups do not share the same USR.
static void testOffsetGroups(const char *Code,
                             const std::vector<std::vector<unsigned>> Groups) {
  std::set<std::string> AllUSRs, CurrUSR;

  for (const auto &Group : Groups) {
    // Groups the invariants do not hold then the value of USR is also invalid,
    // but at that point the test has already failed and USR ceases to be
    // useful.
    std::string USR;
    for (const auto &Offset : Group) {
      USRFindingAction Action(Offset, std::string());
      auto Factory = tooling::newFrontendActionFactory(&Action);
      EXPECT_TRUE(tooling::runToolOnCode(Factory->create(), Code));
      const auto &USRs = Action.getUSRs();
      EXPECT_EQ(1u, USRs.size());
      USR = USRs[0];
      CurrUSR.insert(USR);
    }
    EXPECT_EQ(1u, CurrUSR.size());
    CurrUSR.clear();
    AllUSRs.insert(USR);
  }

  EXPECT_EQ(Groups.size(), AllUSRs.size());
}


TEST(USRLocFinding, FindsVarUSR) {
  const char VarTest[] = "\n\
namespace A {\n\
int foo;\n\
}\n\
int foo;\n\
int bar = foo;\n\
int baz = A::foo;\n\
void fun1() {\n\
  struct {\n\
    int foo;\n\
  } b = { 100 };\n\
  int foo = 100;\n\
  baz = foo;\n\
  {\n\
    extern int foo;\n\
    baz = foo;\n\
    foo = A::foo + baz;\n\
    A::foo = b.foo;\n\
  }\n\
 foo = b.foo;\n\
}\n";
  std::vector<std::vector<unsigned>> VarTestOffsets(3);
  VarTestOffsets[0].push_back(19);
  VarTestOffsets[0].push_back(63);
  VarTestOffsets[0].push_back(205);
  VarTestOffsets[0].push_back(223);
  VarTestOffsets[1].push_back(30);
  VarTestOffsets[1].push_back(45);
  VarTestOffsets[1].push_back(172);
  VarTestOffsets[1].push_back(187);
  VarTestOffsets[2].push_back(129);
  VarTestOffsets[2].push_back(148);
  VarTestOffsets[2].push_back(242);

  testOffsetGroups(VarTest, VarTestOffsets);
}

} // namespace test
} // namespace rename
} // namespace clang
