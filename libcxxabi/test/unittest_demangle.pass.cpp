//===----------------------- unittest_demangle.cpp ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include "../src/cxa_demangle.cpp"

using namespace __cxxabiv1;

void testPODSmallVector() {
  { // {push/pop}_back
    PODSmallVector<int, 1> PSV;
    PSV.push_back(0);
    PSV.push_back(1);
    PSV.push_back(2);
    PSV.push_back(3);
    for (int i = 0; i < 4; ++i)
      assert(PSV[i] == i);
    PSV.pop_back();
    for (int i = 0; i < 3; ++i)
      assert(PSV[i] == i);
    PSV.pop_back();
    PSV.pop_back();
    assert(!PSV.empty() && PSV.size() == 1);
    PSV.pop_back();
    assert(PSV.empty() && PSV.size() == 0);
  }

  {
    PODSmallVector<int, 1> PSV1;
    PSV1.push_back(1);
    PSV1.push_back(2);
    PSV1.push_back(3);

    PODSmallVector<int, 1> PSV2;
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 0);
    assert(PSV2.size() == 3);
    int i = 1;
    for (int x : PSV2) {
      assert(x == i);
      ++i;
    }
    assert(i == 4);
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 3);
    assert(PSV2.size() == 0);
    i = 1;
    for (int x : PSV1) {
      assert(x == i);
      ++i;
    }
    assert(i == 4);
  }

  {
    PODSmallVector<int, 10> PSV1;
    PODSmallVector<int, 10> PSV2;
    PSV1.push_back(0);
    PSV1.push_back(1);
    PSV1.push_back(2);
    assert(PSV1.size() == 3);
    assert(PSV2.size() == 0);
    std::swap(PSV1, PSV2);
    assert(PSV1.size() == 0);
    assert(PSV2.size() == 3);
    int i = 0;
    for (int x : PSV2) {
      assert(x == i);
      ++i;
    }
    for (int x : PSV1) {
      assert(false);
      (void)x;
    }
  }
}

void testSubstitutionTable() {
  {
    SubstitutionTable<2> Tab;

    NameType Names[] = {{"MERP"}, {"MARP"}, {"MAMP"}};
    Tab.pushPack();
    Tab.pushSubstitutionIntoPack(&Names[0]);
    Tab.pushSubstitutionIntoPack(&Names[1]);
    Tab.pushSubstitutionIntoPack(&Names[2]);

    int Index = 0;
    for (Node* N : Tab.nthSubstitution(0)) {
      assert(static_cast<NameType*>(N)->getName() == Names[Index].getName());
      ++Index;
    }
    assert(Index == 3);

    Tab.popPack();
    assert(Tab.empty() && Tab.size() == 0);
    Tab.pushSubstitution(&Names[0]);
    Tab.pushSubstitution(&Names[1]);
    assert(!Tab.empty() && Tab.size() == 2);

    int I = 0;
    for (Node* N : Tab.nthSubstitution(0)) {
      assert(static_cast<NameType*>(N)->getName() == "MERP");
      assert(I == 0);
      ++I;
    }
    for (Node* N : Tab.nthSubstitution(1)) {
      assert(static_cast<NameType*>(N)->getName() == "MARP");
      assert(I == 1);
      ++I;
    }
  }
}

int main() {
  testPODSmallVector();
  testSubstitutionTable();
}
